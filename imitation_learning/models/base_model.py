import os
import torch
import torch.nn as nn
from imitation_learning.utils.general_utils import AttrDict
import sys
if sys.version_info[0] == 2:
    import cPickle as pkl
else:
    import pickle as pkl
from imitation_learning.utils.general_utils import Configurable
from imitation_learning.utils.general_utils import move_to_device
# from imitation_learning.models.gcbc_transfer import GCBCTransfer

class BaseModel(nn.Module, Configurable):
    def __init__(self, override_params, logger):
        super(BaseModel, self).__init__()
        self._hp = self._default_hparams()
        self._override_defaults(override_params)
        self._logger = logger

        self.normalizing_params = None
        if self._hp.dataset_normalization:
            if self._hp.store_normalization_inmodel:
                self.setup_normalizing_params()
            else:
                self.load_normalizing_params()

        self.throttle_log_images = 0

    def set_dataset_sufix(self, hp):
        self.dataset_sufix = hp.name

    def load_normalizing_params(self):
        if 'single_task' in self._hp.data_conf:  # skip if using GCBCTransfer model
            return

        if 'dataset0' in self._hp.data_conf: # used for RandomMixingDataset
            params_dir = self._hp.data_conf.dataset0[1]['data_dir']
            if isinstance(params_dir, list):
                params_dir = params_dir[0]
        elif self._hp.normalizing_params is not None:
            params_dir = self._hp.normalizing_params
        else:
            # when using a list of data_dirs in the single-dataset loader
            if isinstance(self._hp.data_conf['data_dir'], list):
                params_dir = self._hp.data_conf['data_dir'][0]
            else:
                params_dir = self._hp.data_conf['data_dir']
        print('getting normalizing params from ', params_dir)
        dict = pkl.load(open(params_dir + '/normalizing_params.pkl', "rb"))
        self.normalizing_params = move_to_device(dict, self._hp.device)

    def setup_normalizing_params(self):
        self.states_mean = nn.Parameter(torch.tensor(torch.zeros(self._hp.state_dim), dtype=torch.float32))
        self.states_std = nn.Parameter(torch.tensor(torch.zeros(self._hp.state_dim), dtype=torch.float32))
        self.actions_mean = nn.Parameter(torch.tensor(torch.zeros(self._hp.action_dim), dtype=torch.float32))
        self.actions_std = nn.Parameter(torch.tensor(torch.zeros(self._hp.action_dim), dtype=torch.float32))

    def set_normalizing_params(self, dict):
        for k, v in dict.items():
            setattr(self, k, nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=False))

    def _default_hparams(self):
        # General Params:
        default_dict = AttrDict({
            'batch_size': -1,
            'max_seq_len': -1,
            'device':torch.device('cuda'),
            'data_conf':None,
            'restore_path':None,
            'dataset_normalization':True,  # path to pkl file with normalization parameters
            'store_normalization_inmodel':True,
            'normalizing_params': None,
            'phase': None,
            'stage': 'main'  # or finetuning
        })
        
        # Network params
        default_dict.update({
            'normalization': 'batch',
        })

        # add new params to parent params
        return AttrDict(default_dict)


    def build_network(self):
        raise NotImplementedError("Need to implement this function in the subclass!")

    def forward(self, inputs):
        raise NotImplementedError("Need to implement this function in the subclass!")

    def loss(self, model_inputs, model_output):
        raise NotImplementedError("Need to implement this function in the subclass!")

    def apply_dataset_normalization(self, tensor, name):
        """
        :param tensor:
        :param name: either 'states' or 'actions'
        :return:
        """
        return (tensor - self.__getattr__(name + '_mean')) / (self.__getattr__(name + '_std') + 1e-6)

    def unnormalize_dataset(self, tensor, name):
        """
        :param tensor:
        :param name: either 'states' or 'actions'
        :return:
        """
        return tensor * self.__getattr__(name + '_std') + self.__getattr__(name + '_mean')

    def log_outputs(self, model_output, inputs, losses, step, phase):
        # Log generally useful outputs
        self._log_losses(losses, step)

        # if phase == 'train':
        #     self.log_gradients(step, phase)

        if self.throttle_log_images % 10 == 0:
            self.throttle_log_images = 0
            for module in self.modules():
                if hasattr(module, '_log_outputs'):
                    module._log_outputs(model_output, inputs, losses, step, phase)
        self.throttle_log_images += 1

    def _log_losses(self, losses, step):
        for name, loss in losses.items():
            name += "_" + self.dataset_sufix
            if torch.is_tensor(loss):
                self._logger.log_scalar(loss, name, step)
            else:
                self._logger.log_scalar(loss[0], name, step)

    def _restore_params(self, strict=True):
        checkpoint = torch.load(self._hp.restore_path, map_location=self._hp.device)
        print('restoring parameters from ', self._hp.restore_path)
        self.load_state_dict(checkpoint['state_dict'], strict=strict)

    def _load_weights(self, weight_loading_info):
        """
        Loads weights of submodels from defined checkpoints + scopes.
        :param weight_loading_info: list of tuples: [(model_handle, scope, checkpoint_path)]
        """

        def get_filtered_weight_dict(checkpoint_path, scope):
            if os.path.isfile(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self._hp.device)
                filtered_state_dict = {}
                remove_key_length = len(scope) + 1      # need to remove scope from checkpoint key
                for key, item in checkpoint['state_dict'].items():
                    if key.startswith(scope):
                        filtered_state_dict[key[remove_key_length:]] = item
                if not filtered_state_dict:
                    raise ValueError("No variable with scope '{}' found in checkpoint '{}'!".format(scope, checkpoint_path))
                return filtered_state_dict
            else:
                raise ValueError("Cannot find checkpoint file '{}' for loading '{}'.".format(checkpoint_path, scope))

        print("")
        for loading_op in weight_loading_info:
            print(("=> loading '{}' from checkpoint '{}'".format(loading_op[1], loading_op[2])))
            filtered_weight_dict = get_filtered_weight_dict(checkpoint_path=loading_op[2],
                                                            scope=loading_op[1])
            loading_op[0].load_state_dict(filtered_weight_dict)
            print(("=> loaded '{}' from checkpoint '{}'".format(loading_op[1], loading_op[2])))
        print("")

    def log_gradients(self, step, phase):
        grad_norms = list([torch.norm(p.grad.data) for p in self.parameters() if p.grad is not None])
        grad_names = list([name for name, p in self.named_parameters() if p.requires_grad])

        if len(grad_norms) == 0:
            return
        grad_norms = torch.stack(grad_norms)

        for name, grad_norm in zip(grad_names, grad_norms):
            self._logger.log_scalar(grad_norm.mean(), 'gradients/{}mean_norm'.format(name), step, phase)
            self._logger.log_scalar(grad_norm.max(), 'gradients/{}max_norm'.format(name), step, phase)

        self._logger.log_scalar(grad_norms.mean(), 'gradients/mean_norm', step, phase)
        self._logger.log_scalar(grad_norms.max(), 'gradients/max_norm', step, phase)

