import matplotlib;
import torch
import  imp

matplotlib.use('Agg');
import inspect
import copy
import glob
import json
import argparse
import os
import time
import datetime
import numpy as np
from torch import autograd
from torch.optim import Adam
from functools import partial
from imitation_learning.utils.general_utils import move_to_device
from imitation_learning.utils.general_utils import map_recursive
from imitation_learning.utils.general_utils import AverageMeter, RecursiveAverageMeter
from imitation_learning.utils.general_utils import AttrDict
from imitation_learning.utils.checkpointer import CheckpointHandler
from imitation_learning.utils.tensorboard_logger import Logger
from imitation_learning.models.gcbc_transfer import GCBCTransfer
from imitation_learning.models import get_model_class
from imitation_learning.data_sets import get_dataset_class

from imitation_learning.models.utils.compute_dataset_normalization import compute_dataset_normalization
from imitation_learning.utils.general_utils import sorted_nicely
import shutil

def save_checkpoint(state, folder, filename='checkpoint.pth'):
    print('saving checkpoint ', os.path.join(folder, filename))
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
    return os.path.join(folder, filename)

def delete_older_checkpoints(path):
    files = glob.glob(path + '/weights_itr*.pth')
    files = sorted_nicely(files)
    if len(files) > 1:
        for f in files[:-1]:
            os.remove(f)
    files = glob.glob(path + '/weights_best*.pth')
    files = sorted_nicely(files)
    if len(files) > 1:
        for f in files[:-1]:
            os.remove(f)

def clear_folder(path):
    if os.path.exists(path + '/weights'):
        shutil.rmtree(path + '/weights')
    if os.path.exists(path + '/events'):
        shutil.rmtree(path + '/events')
    for f in glob.glob( path + "/*.json"):
        os.remove(f)

def datetime_str():
    return datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")

def make_path(exp_dir, conf_path, prefix, make_new_dir):
    # extract the subfolder structure from config path
    if conf_path.endswith('.json'):
        return '/'.join(str.split(conf_path, '/')[:-1])
    else:
        path = conf_path.split('imitation_learning_experiments/', 1)[1]
        if make_new_dir:
            prefix += datetime_str()
        base_path = os.path.join(exp_dir,  '/'.join(str.split(path, '/')[:-1]))
        return os.path.join(base_path, prefix) if prefix else base_path

def set_seeds(seed):
    """Sets all seeds and disables non-determinism in cuDNN backend."""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def run_control_eval(logger, control_conf, model_weights_path, global_step):
    from visual_mpc.agent.trajectory_collector import TrajectoryCollector
    control_conf = copy.deepcopy(control_conf)

    repeats = 1 # rerun evaluation multiple times
    for label, conf in control_conf.items():
        success_rates = []
        for r in range(repeats):
            conf['logging_conf'] = AttrDict(logger=logger, label=label + '_r{}'.format(r), global_step=global_step)
            conf['policy']['restore_path'] = model_weights_path
            col = TrajectoryCollector(conf)
            col.run()
            success_rates.append(col.metrics[label + '_r{}'.format(r) + '_goal_reached_mean'])
        logger.log_scalar(np.std(success_rates), label + '_success_rate_std', global_step)
        logger.log_scalar(np.mean(success_rates), label + '_success_rate_mean', global_step)


from imitation_learning.utils.general_utils import Configurable

class ModelTrainer(Configurable):
    def __init__(self, args):
        import wandb
        from imitation_learning.config import WANDB_API_KEY, WANDB_EMAIL, WANDB_USERNAME
        os.environ['WANDB_API_KEY'] = WANDB_API_KEY
        os.environ['WANDB_USER_EMAIL'] = WANDB_EMAIL
        os.environ['WANDB_USERNAME'] = WANDB_USERNAME
        os.environ["WANDB_MODE"] = "run"
        wandb.init(project='bridge_data', reinit=True, sync_tensorboard=True, name=args.prefix)
        self.batch_idx = 0
        self.args = args

        ## Set up params
        self.conf, self.model_conf, self.data_conf = self.get_configs()
        if args.data_config_override is not None:
            if 'data_dir' in args.data_config_override:
                override_dict = {'data_dir': os.environ['DATA'] + '/' + args.data_config_override['data_dir'],
                                 'name': '_'.join(str.split(args.data_config_override['data_dir'], '/')[-2:])}
            else:
                override_dict = args.data_config_override
            if 'main' in self.data_conf:
                self.data_conf['main']['dataconf'].update(override_dict)
            if 'finetuning' in self.data_conf:
                self.data_conf['finetuning']['dataconf'].update(override_dict)
                self.data_conf['finetuning']['val0']['dataconf'].update(override_dict)
            print('data_conf after override: ', self.data_conf)
        elif args.source_data_config_override is not None:
            if 'data_dir' in args.source_data_config_override:
                override_dict = {'data_dir': os.environ['DATA'] + '/' + args.source_data_config_override['data_dir'],
                                 'name': '_'.join(str.split(args.source_data_config_override['data_dir'], '/')[-2:])}
            else:
                override_dict = args.source_data_config_override
            self.data_conf.main.dataconf.dataset0[1].update(override_dict)
            self.data_conf.main.val0.dataconf.update(override_dict)
            print('data_conf after override: ', self.data_conf)
        self._loggers = AttrDict()

        if args.gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

        self._hp = self._default_hparams()
        self._override_defaults(self.conf)  # override defaults with config file

        self._hp.exp_path = make_path(os.environ['EXP'] + '/spt_experiments', args.path, args.prefix, args.new_dir)
        if not args.resume:
            clear_folder(self._hp.exp_path)
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, 'events')
        print('using log dir: ', log_dir)

        self.run_testmetrics = args.metric
        if args.deterministic:
            set_seeds(args.deterministic)

        if args.cpu:
            self.device = torch.device('cpu')
            self.use_cuda = False
        else:
            self.use_cuda = torch.cuda.is_available()
            self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        if not args.skip_main:  # if resuming finetuning skip the mainstage
            self.run_stage(args, 'main')
        if self._hp['finetuning']:
            self.run_stage(args, 'finetuning')

    def run_stage(self, args, stage):
        print('preparing stage ', stage)
        self.stage = stage
        reuse_action_predictor = True
        if stage == 'finetuning' and not args.resume:  # if resuming finetuning we don't initialize from mainstage
            if self._hp['main'].model is GCBCTransfer:
                trained_model_state_dict = copy.deepcopy(self.model.single_task_model.state_dict())
            else:
                trained_model_state_dict = copy.deepcopy(self.model.state_dict())

            if 'goal_cond' in self.model_conf['main'] and 'goal_cond' not in self.model_conf['finetuning']:
                reuse_action_predictor = False
                for key in copy.deepcopy(trained_model_state_dict):
                    if 'action_predictor' in key:
                        trained_model_state_dict.pop(key)

        if stage != 'main':
            self._hp.exp_path = os.path.join(self._hp.exp_path, stage)
        print('Writing to the experiment directory: {}'.format(self._hp.exp_path))
        if not os.path.exists(self._hp.exp_path):
            os.makedirs(self._hp.exp_path)
        self.save_dir = os.path.join(self._hp.exp_path, 'weights')

        dataset_class = self.data_conf[stage].dataclass
        data_conf = self.data_conf[stage].dataconf

        model_class = self._hp[self.stage].model
        self.model, self.train_loader, self.train_dataset = self.make_model_and_dataset(self._hp.logger, model_class,
                                                                                        dataset_class, data_conf, 'train', stage)
        self.model_val, val_loader, val_dataset = self.make_model_and_dataset(self._hp.logger, model_class,
                                                                                        dataset_class, data_conf, 'val', stage)
        if self._hp.dataset_normalization:
                self.model.set_normalizing_params(self.norm_params)
                self.model_val.set_normalizing_params(self.norm_params)

        self.val_loaders = [val_loader]
        self.val_datasets = [val_dataset]
        self.make_additional_valdatasets(stage)

        # to make sure train and validation loaders are using the same task and domain indices
        all_datasets = [self.train_dataset, *self.val_datasets]
        all_domains = set([d for dataset in all_datasets for d in list(dataset.domain_hash_index.keys())])
        all_taskdescriptions = set([d for dataset in all_datasets for d in dataset.taskdescription2task_index.keys()])
        domain_hash_index = {domain_hash: index for domain_hash, index in
                                               zip(all_domains, range(len(all_domains)))}
        if args.load_task_indices:
            with open(args.load_task_indices) as f:
                taskdescription2task_index = json.load(f)
        else:
            taskdescription2task_index = {task_descp: index for task_descp, index in
                                               zip(all_taskdescriptions, range(len(all_taskdescriptions)))}
        for dataset in all_datasets:
            dataset.set_domain_and_taskdescription_indices(domain_hash_index, taskdescription2task_index)
        
        with open(os.path.join(self._hp.exp_path, "task_index.json"), 'w') as f:
            json.dump(taskdescription2task_index, f, indent=4)
        
        for dataset in all_datasets:
            with open(os.path.join(self._hp.exp_path, "{}_{}_dataset_stats.txt".format(dataset._hp.name, dataset.phase)), 'w') as f:
                f.write(dataset.dataset_stats)

        self.optimizer = Adam(self.model.parameters(), lr=self._hp.lr, weight_decay=self._hp.weight_decay)

        self._hp.mpar = self.model._hp
        save_config({'train._hp': self._hp, 'model_conf': self.model._hp, 'data_conf': self.data_conf},
                    os.path.join(self._hp.exp_path, stage + "_conf" + datetime_str() + ".json"))

        if stage.startswith('finetuning') and not args.resume:  # if resuming finetuning skip the mainstage
            incompatible_keys = self.model.load_state_dict(trained_model_state_dict, strict=False)
            if reuse_action_predictor:
                assert len(incompatible_keys.missing_keys) == 0
            else:
                print('Warning, missing keys in stage{}: {}'.format(stage, incompatible_keys.missing_keys))
            print('Warning, unexpected keys in stage{}: {}'.format(stage, incompatible_keys.unexpected_keys))
        if args.resume or self._hp.resume_checkpoint is not None:
            if self._hp.resume_checkpoint is not None:
                args.resume = self._hp.resume_checkpoint
            start_epoch, self.global_step = self.resume(args.resume)
            if stage.startswith('finetuning') and 'finetuning' not in args.resume:
                start_epoch = 0
                self.global_step = 0
            args.resume = False # avoid having it crash in the finetuning stage if we resume in the main stage
        else:
            self.global_step = 0
            start_epoch = 0
        if 'num_epochs' in self._hp[self.stage]:
            num_epochs = self._hp[self.stage].num_epochs
            max_iterations = None
        else:
            num_epochs = None
            max_iterations = self._hp[self.stage].max_iterations
        self.best_val_loss = float('inf')
        self.train(start_epoch, num_epochs, max_iterations)

    def make_additional_valdatasets(self, stage):
        val_keys = [k for k in self.data_conf[stage].keys() if k.startswith('val')]
        if val_keys == []:
            return
        for key in val_keys:
            print('making extra val dataset with key: ', key)
            dataclass = self.data_conf[stage][key].dataclass
            dataconf = self.data_conf[stage][key].dataconf
            dataset = dataclass(dataconf, 'val', shuffle=True)
            loader = dataset.get_data_loader(self._hp.batch_size)
            self.val_datasets.append(dataset)
            self.val_loaders.append(loader)

    def make_model_and_dataset(self, logger, ModelClass, DatasetClass, data_conf, phase, stage):
        logger = logger(os.path.join(self.log_dir, phase))
        self._loggers[phase] = logger
        model_conf = copy.deepcopy(self.model_conf)
        if 'main' in model_conf:
            model_conf = model_conf[stage]
        model_conf['batch_size'] = self._hp.batch_size
        model_conf['device'] = self.device.type
        if stage == 'finetuning':
            if 'finetuning_override' in model_conf:
                model_conf.update(self.model_conf.finetuning_override)
            model_conf['stage'] = 'finetuning'
        model_conf['phase'] = phase
        model_conf.data_conf = data_conf
        model = ModelClass(model_conf, logger)
        model.to(self.device)
        model.device = self.device
        if phase is not 'test':
            dataset = DatasetClass(data_conf, phase, shuffle=True)
            loader = dataset.get_data_loader(self._hp.batch_size)

        if phase == 'train' and self._hp.dataset_normalization:
            self.norm_params = compute_dataset_normalization(loader)

        return model, loader, dataset

    def _default_hparams(self):
        # put new parameters in here:
        default_dict = {
            'resume_checkpoint': None,
            'logger': Logger,
            'batch_size': 32,
            'mpar': None,   # model parameters
            'data_conf': None,   # model data parameters
            'exp_path': None,  # Path to the folder with experiments
            'log_every': 1,
            'delete_older_checkpoints': True,
            'epoch_cycles_train': 1,
            'optimizer': 'adam',    # supported: 'adam', 'rmsprop', 'sgd'
            'lr': 1e-4,
            'momentum': 0,      # momentum in RMSProp / SGD optimizer
            'adam_beta': 0.9,       # beta1 param in Adam
            'main': None,
            'finetuning': None,
            'weight_decay': 0,
            'dataset_normalization': True,
            'delta_step_val': 100,
            'delta_step_control_val': 500,
            'delta_step_save': 500,
        }
        # add new params to parent params
        return AttrDict(default_dict)

    def get_configs(self):
        conf_path = os.path.abspath(self.args.path)

        if conf_path.endswith('.json'):
            with open(conf_path, 'r') as f:
                json_conf = json.load(f)
            conf = json_conf['train._hp']
            conf['model'] = get_model_class(conf['model'])
            conf['finetuning_model'] = get_model_class(conf['finetuning_model'])
            conf['dataset_class'] = get_dataset_class(conf['dataset_class'])
            conf['finetuning_dataset_class'] = get_dataset_class(conf['finetuning_dataset_class'])
            conf['identical_default_ok'] = ''
            if conf['logger'] == 'Logger':
                conf['logger'] = Logger
            else:
                raise NotImplementedError
            model_conf = json_conf['model_conf']
            model_conf['identical_default_ok'] = ''
            data_conf = json_conf['data_conf']
            data_conf['identical_default_ok'] = ''
        else:
            print('loading from the config file {}'.format(conf_path))
            conf_module = imp.load_source('conf', self.args.path)
            conf = conf_module.configuration
            model_conf = conf_module.model_config
            data_conf = conf_module.data_config

        return conf, model_conf, data_conf

    def resume(self, ckpt):
        weights_file = CheckpointHandler.get_resume_ckpt_file(ckpt, os.path.join(self._hp.exp_path, 'weights'))
        global_step, start_epoch, _ = \
            CheckpointHandler.load_weights(weights_file, self.model,
                                           load_step_and_opt=True, optimizer=self.optimizer,
                                           dataset_length=len(self.train_loader) * self._hp.batch_size,
                                           strict=self.args.strict_weight_loading)
        self.model.to(self.model.device)
        return start_epoch, global_step

    def train(self, start_epoch, num_epochs, max_iterations):
        if max_iterations is not None:
            num_epochs = int(np.ceil(max_iterations/len(self.train_loader)))
            print('setting num_epochs to ', num_epochs)
        self.last_global_step_when_val = int(-1e9) # make sure we do val the first time of the train-val cycle.
        self.last_global_step_when_control = int(-1e9)
        self.last_global_step_when_save = int(-1e9)
        for i, epoch in enumerate(range(start_epoch, num_epochs)):
            self.train_val_cycle(epoch, num_epochs)
        return epoch

    def train_val_cycle(self, epoch, num_epochs):
        if not self.args.no_train:
            self.train_epoch(epoch, num_epochs)
        if (self.global_step - self.last_global_step_when_val) > self._hp.delta_step_val and not self.args.no_val:
            val_loss = self.val()
            self.last_global_step_when_val = self.global_step
        if (self.global_step - self.last_global_step_when_save) > self._hp.delta_step_save or self.run_control_now():
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_string = 'weights_best_itr{}.pth'.format(self.global_step)
            else:
                save_string = 'weights_itr{}.pth'.format(self.global_step)
            self._save_file_name = save_checkpoint({
                    'epoch': epoch,
                    'global_step': self.global_step,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, self.save_dir, save_string)
            if self._hp.delete_older_checkpoints:
                delete_older_checkpoints(self.save_dir)
            self.last_global_step_when_save = self.global_step
        if self.run_control_now():
            run_control_eval(self._loggers.val, self._hp[self.stage].control_conf, self._save_file_name, self.global_step)
            self.last_global_step_when_control = self.global_step

    def run_control_now(self):
        return 'control_conf' in self._hp[self.stage] and (self.global_step - self.last_global_step_when_control) > self._hp.delta_step_control_val and not self.args.no_val

    @property
    def log_outputs_now(self):
        return self.global_step % self.log_outputs_interval == 0

    def train_epoch(self, epoch, num_epochs):
        self.model.train()
        self.model.to(self.device)
        epoch_len = len(self.train_loader)
        end = time.time()
        batch_time = AverageMeter()
        upto_log_time = AverageMeter()
        data_load_time = AverageMeter()
        self.log_outputs_interval = 50
        print('starting epoch ', epoch)
        self.model.set_dataset_sufix(self.train_dataset._hp)

        for self.batch_idx, sample_batched in enumerate(self.train_loader):
            data_load_time.update(time.time() - end)
            inputs = move_to_device(sample_batched, self.device)
            self.optimizer.zero_grad()
            inputs.global_step = self.global_step
            inputs.max_iterations = self._hp[self.stage].max_iterations
            output = self.model(inputs)
            losses = self.model.loss(inputs, output)
            losses.total_loss.backward()
            self.optimizer.step()

            upto_log_time.update(time.time() - end)
            if self.log_outputs_now:
                self.model.log_outputs(output, inputs, losses, self.global_step,
                                       phase='train')
            batch_time.update(time.time() - end)

            if self.log_outputs_now:
                print('GPU {}: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"] if self.use_cuda else 'none', self._hp.exp_path))
                print(('stage {}, itr: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(self.stage,
                    self.global_step, epoch, self.batch_idx, len(self.train_loader),
                    100. * self.batch_idx / len(self.train_loader), losses.total_loss.item())))
                
                print('avg time for loading: {:.2f}s, logs: {:.2f}s, compute: {:.2f}s, total: {:.2f}s'
                      .format(data_load_time.avg,
                              batch_time.avg - upto_log_time.avg,
                              upto_log_time.avg - data_load_time.avg,
                              batch_time.avg))
                togo_train_time = batch_time.avg * (num_epochs - epoch) * epoch_len / 3600.
                print('ETA: {:.2f}h'.format(togo_train_time))
            
            del output, losses
            self.global_step = self.global_step + 1
            end = time.time()

            if self.global_step > self._hp[self.stage].max_iterations:
                break

        self.model.to(torch.device('cpu'))

    def val(self):
        val_losses = []
        for val_loader, val_dataset in zip(self.val_loaders, self.val_datasets):
            val_loss = self.eval_for_dataset(val_loader, val_dataset)
            val_losses.append(val_loss)
        return val_losses[0]

    def eval_for_dataset(self, val_loader, val_dataset):
        print('Running Testing')
        start = time.time()
        self.model_val.to(self.device)
        self.model_val.load_state_dict(self.model.state_dict())
        self.model_val.eval()
        self.model_val.throttle_log_images = 0  # make sure to log images the first val pass!
        self.model_val.set_dataset_sufix(val_dataset._hp)
        losses_meter = RecursiveAverageMeter()
        with autograd.no_grad():
            for batch_idx, sample_batched in enumerate(val_loader):
                inputs = move_to_device(sample_batched, self.device)
                inputs.global_step = self.global_step
                inputs.max_iterations = self._hp[self.stage].max_iterations
                output = self.model_val(inputs)
                losses = self.model_val.loss(inputs, output)
                losses_meter.update(losses)
                del losses

            self.model_val.log_outputs(
                output, inputs, losses_meter.avg, self.global_step, phase='val')
            print(('\nTest set: Average loss: {:.4f} in {:.2f}s over {} batches\n'
                   .format(losses_meter.avg.total_loss.item(), time.time() - start, batch_idx)))
        del output
        self.model_val.to(torch.device('cpu'))
        return losses_meter.avg.total_loss.item()

    def get_optimizer_class(self):
        if self._hp.optimizer == 'adam':
            optim = partial(Adam, betas=(self._hp.adam_beta, 0.999))
        else:
            raise ValueError("Optimizer '{}' not supported!".format(self._hp.optimizer))
        return optim


def save_config(confs, exp_conf_path):
    def func(x):
        if inspect.isclass(x) or inspect.isfunction(x):
            return x.__name__
        else:
            return x
    confs = map_recursive(func, confs)

    with open(exp_conf_path, 'w') as f:
        json.dump(confs, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to the config file directory")
    # Folder settings
    parser.add_argument("--prefix", help="experiment prefix, if given creates subfolder in experiment directory")
    parser.add_argument('--new_dir', default=False, action='store_true', help='If True, concat datetime string to exp_dir.')

    # Running protocol
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--skip_main', default=False, action='store_true',
                        help='if true will go directly to fine-tuning stage')
    parser.add_argument('--no_train', default=False, action='store_true',
                        help='if False will not run training epoch')
    parser.add_argument('--no_val', default=False, action='store_true',
                        help='if False will not run validation epoch')
    parser.add_argument('--metric', default=False, action='store_true',
                        help='if True, run test metrics')
    parser.add_argument('--cpu', default=False,
                        help='if True, use CPU', action='store_true')

    # Misc
    parser.add_argument('--gpu', default=-1, type=int,
                        help='will set CUDA_VISIBLE_DEVICES to selected value')
    parser.add_argument('--strict_weight_loading', default=True, type=int,
                        help='if True, uses strict weight loading function')
    parser.add_argument('--deterministic', default=False, action='store_true',
                        help='if True, sets fixed seeds for torch and numpy')
    parser.add_argument('--data_config_override', default=None,  help='used in dooodad for sweeps')
    parser.add_argument('--load_task_indices', default=None,  help='task indices json file to load')
    parser.add_argument('--source_data_config_override', default=None,  help='used in dooodad for sweeps')
    args = parser.parse_args()
    ModelTrainer(args)
