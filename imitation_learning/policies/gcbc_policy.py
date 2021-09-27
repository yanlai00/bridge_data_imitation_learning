import numpy as np
import torch
import os
from widowx_envs.policies.policy import Policy
import cv2
import glob
from imitation_learning.utils.general_utils import AttrDict
from imitation_learning.utils.general_utils import np_unstack
import json

from widowx_envs.utils.datautils.raw2lmdb import crop_image

class BCPolicyStates(Policy):
    """
    Behavioral Cloning Policy
    """
    def __init__(self, ag_params, policyparams):
        super(BCPolicyStates, self).__init__()
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)

        model, model_config = self.get_saved_params(policyparams)
        model_config['batch_size'] = 1
        model_config['restore_path'] = self._hp.restore_path

        self.predictor = model(model_config)
        self.predictor.eval()
        self.device = torch.device('cuda')
        self.predictor.to(self.device)
        print('finished setting up policy')


    def get_saved_params(self, policyparams):
        if str.split(policyparams['restore_path'], '/')[-3] == 'finetuning':
            search_pattern = '/finetuning_conf*'
            stage = 'finetuning'
        else:
            search_pattern = '/main_conf*'
            stage = 'main'
        search_pattern = '/'.join(str.split(policyparams['restore_path'], '/')[:-2]) + search_pattern
        conffile = glob.glob(search_pattern)
        if len(conffile) == 0:
            raise ValueError('no conf files found in ', search_pattern)
        conffile = conffile[0]
        with open(conffile, 'r') as f:
            conf = json.load(f)
        if conf['train._hp'][stage]['model'] == 'GCBCImages':
            from imitation_learning.models.gcbc_images import GCBCImagesModelTest
            model = GCBCImagesModelTest
            new_conf = conf['model_conf']
        elif conf['train._hp'][stage]['model'] == 'GCBCTransfer':
            from imitation_learning.models.gcbc_images import GCBCImagesModelTest
            model = GCBCImagesModelTest
            new_conf = conf['model_conf']['shared_params']
            new_conf.update(conf['model_conf'][self._hp.get_sub_model])
            new_conf['get_sub_model'] = self._hp.get_sub_model
        else:
            raise ValueError('model not found!')
        new_conf['identical_default_ok'] = ''
        new_conf.update(self._hp.model_override_params)
        return model, new_conf

    def reset(self):
        super(BCPolicyStates, self).reset()

    def _default_hparams(self):
        default_dict = AttrDict({
            'restore_path': None,
            'verbose': False,
            'type': None,
            'model_override_params': None,
            'get_sub_model': 'single_task_params',
        })
        default_dict.update(super(BCPolicyStates, self)._default_hparams())
        return default_dict

    def act(self, t=None, i_tr=None, state=None, loaded_traj_info=None):
        self.t = t
        self.i_tr = i_tr
        goal_states = loaded_traj_info['state'][-1]

        inputs = AttrDict(state=self.npy2trch(state[-1][None]),
                          goal_state=self.npy2trch(goal_states[None]))
        out = self.predictor(inputs)

        output = AttrDict()
        output.actions = out['a_pred'].data.cpu().numpy()[0]
        return output

    @property
    def default_action(self):
        return np.zeros(self.predictor._hp.n_actions)

    def log_outputs_stateful(self, logger=None, global_step=None, phase=None, dump_dir=None, exec_seq=None, goal=None, index=None, env=None, goal_pos=None, traj=None, topdown_image=None):
        logger.log_video(np.transpose(exec_seq, [0, 3, 1, 2]), 'control/traj{}_'.format(index), global_step, phase)
        goal_img = np.transpose(goal, [2, 0, 1])[None]
        goal_img = torch.tensor(goal_img)
        logger.log_images(goal_img, 'control/traj{}_goal'.format(index), global_step, phase)

    def npy2trch(self, arr):
        return torch.from_numpy(arr).float().to(self.device)

class GCBCPolicyImages(BCPolicyStates):
    def __init__(self, ag_params, policyparams):
        super(GCBCPolicyImages, self).__init__(ag_params, policyparams)
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)

    def _default_hparams(self):
        default_dict = AttrDict({
            'confirm_first_image': False,
            'crop_image_region': False,
            'stack_goal_images': False,
        })
        default_dict.update(super(GCBCPolicyImages, self)._default_hparams())
        return default_dict

    @staticmethod
    def _preprocess_input(input):
        assert len(input.shape) == 4    # can currently only handle inputs with 4 dims
        if input.max() > 1.0:
            input = input / 255.
        if input.min() >= 0.0:
            input = 2*input - 1.0
        if input.shape[-1] == 3:
            input = input.transpose(0, 3, 1, 2)
        return input

    def act(self, t=None, i_tr=None, images=None, state=None, goal=None, goal_image=None):
        # Note: goal_image provides n (2) images starting from the last images of the trajectory
        self.t = t
        self.i_tr = i_tr
        self.goal_image = goal_image


        images = images[t]
        if self._hp.crop_image_region:
            target_height, target_width = self._hp.model_override_params['data_conf']['image_size_beforecrop']
            if self._hp.crop_image_region == 'select':
                from widowx_envs.utils.datautils.annotate_object_pos import Getdesig
                if self.t == 0:
                    self.crop_center = np.array(Getdesig(images[0]).desig, dtype=np.int32)
                print('selected position', self.crop_center)
            else:
                self.crop_center = self._hp.crop_image_region
            images = crop_image(target_height, target_width, self.crop_center, images)

        if self._hp.model_override_params['data_conf']['image_size_beforecrop'] != images.shape[2:4]:
            h, w = self._hp.model_override_params['data_conf']['image_size_beforecrop']
            resized_images = np.zeros([images.shape[0], h, w, 3], dtype=images.dtype)
            for n in range(images.shape[0]):
                resized_images[n] = cv2.resize(images[n], (w, h), interpolation=cv2.INTER_AREA)
            images = resized_images

        if t == 0 and self._hp.confirm_first_image:
            import matplotlib.pyplot as plt
            import matplotlib
            # matplotlib.use('TkAgg')
            plt.switch_backend('Tkagg')
            # matplotlib.use('Agg')
            if self.predictor._hp.concatenate_cameras:
                plt.imshow(np.concatenate(np_unstack(images, axis=0), 0))
            else:
                plt.imshow(images[self.predictor._hp.sel_camera])
            print('saving start image to', self.traj_log_dir + '/start_image.png')
            # plt.savefig(self.traj_log_dir + '/start_image.png')
            plt.show()
            if self.predictor._hp.goal_cond:
                if self._hp.stack_goal_images:
                    for goal_image_single in goal_image:
                        plt.imshow(goal_image_single[0].transpose(1, 2, 0))
                        plt.show()
                else:
                    plt.imshow(goal_image[0, self.predictor._hp.sel_camera])
                    # plt.savefig(self.traj_log_dir + '/goal_image.png')
                    plt.show()

        images = self.npy2trch(self._preprocess_input(images))

        inputs = AttrDict(I_0=images)
        if self.predictor._hp.goal_cond:
            if self._hp.stack_goal_images:
                inputs['I_g'] = [self.npy2trch(self._preprocess_input(goal_image_single)) for goal_image_single in goal_image]
            else:
                inputs['I_g'] = self.npy2trch(self._preprocess_input(goal_image[-1] if len(goal_image.shape) > 4 else goal_image))

        output = AttrDict()
        action = self.predictor(inputs).pred_actions.data.cpu().numpy().squeeze()
        print('inferred action', action)
        output.actions = action
        return output


if __name__ == '__main__':
    policy = {
        'type': GCBCPolicyImages,
        'restore_path': os.environ['EXP'] + '/spt_experiments' + '/modeltraining/bc/widowx_pushing/can_freeze_pretrain/weights/weights_ep9995.pth',
    }
    p = GCBCPolicyImages({}, policy, None, None)