import ipdb
import torch
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import eval_np
import numpy as np
# from visual_mpc.policy.policy import Policy
from torch import nn
from widowx_envs.policies.policy import Policy
import os
# from visual_mpc.policy.data_augs import random_crop, random_convolution, random_color_jitter
# from rlkit.torch.conv_networks import CNN, ConcatCNN, TwoHeadCNN
from rlkit.torch.conv_networks import ConcatCNN
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.policies.gaussian_policy import GaussianCNNPolicy

import h5py
import pickle

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import imp
import json

from semiparametrictransfer.utils.general_utils import AttrDict

ACTION_LOW = np.array([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0])
ACTION_HIGH = np.array([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1])


class RLPolicyAWAC(Policy):
    def __init__(self, ag_params, policyparams, gpu_id=0):
        self._adim = ag_params['env'][0].adim
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)

        self.enable_gpus(str(gpu_id))
        ptu.set_gpu_mode(True)
        parameters = torch.load(self._hp.path)
        action_dim = 7

        if self._hp.task_id is not None:
            self.task_id_vec = np.zeros((1, self._hp.num_tasks))
            self.task_id_vec[0, self._hp.task_id] = 1
        else:
            self.task_id_vec = None

        conf_path = os.path.abspath(self._hp.exp_conf_path)
        if conf_path.endswith('.py'):
            print('loading from the config file {}'.format(conf_path))
            conf_module = imp.load_source('conf', conf_path)
            variant = conf_module.variant
        elif conf_path.endswith('.json'):
            with open(conf_path, 'r') as f:
                variant = json.load(f)
        else:
            raise ValueError("Invalid experiment conf path!")
        
        cnn_params = variant['cnn_params']

        cnn_params.update(
            added_fc_input_size=self._hp.num_tasks,
        )

        self.policy = GaussianCNNPolicy(
            max_log_std=0,
            min_log_std=-6,
            obs_dim=None,
            action_dim=action_dim,
            std_architecture="values",
            **cnn_params
        )

        self.policy.load_state_dict(parameters['policy_state_dict'])

        if self._hp.load_qfunc:
            cnn_params.update(
                output_size=1,
                added_fc_input_size=action_dim+self._hp.num_tasks,
            )

            self.qf1 = ConcatCNN(**cnn_params)
            self.qf1.load_state_dict(parameters['qf1_state_dict'])

    def enable_gpus(self, gpu_str):
        if gpu_str is not "":
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

    def _default_hparams(self):
        default_dict = {
            'path': None,
            'goal_pos': (0.935024, 0.204873, 0.0694792),
            'data_aug': False,
            'goals_path': '/raid/asap7772/3cam_widowx_data.hdf5',
            'log': False,
            'num_views': 1,
            'goal_cond': False,
            'optimize_q_function': False,
            'exp_conf_path': None,
            'load_qfunc': True,
            'num_tasks': 0,
            'task_id': None,
            'debug': False,
            'normalize': True,
        }
        parent_params = super(RLPolicyAWAC, self)._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def set_log_dir(self, d):
        print('setting log dir')
        super(RLPolicyAWAC, self).set_log_dir(d)

    def multiview_goal_imgs(self):
        rf = h5py.File(self._hp.goals_path, 'r')
        imgs = rf['images'][()]
        imgs1 = rf['images1'][()]
        imgs2 = rf['images2'][()]
        observations = rf['states'][()]
        actions = rf['actions'][()]
        next_imgs = rf['next_images'][()]
        next_observations = rf['next_states'][()]
        next_actions = rf['next_actions'][()]
        index = np.random.randint(0, high=imgs.shape[0]-1, size=1, dtype=int)[0]
        gx,gy,gz = observations[index][:3]
        self.goal_state = observations[index]
        self.goal_pos = (gx,gy,gz)
        print('----------------------')
        print('Goal pos:', gx, gy, gz)
        print('----------------------')
        self.gimages = imgs[index], imgs1[index], imgs2[index]

    def get_qval(self, q, obs, acts):
        if self._hp.task_id is not None:
            return eval_np(q, obs, acts, self.task_id_vec)
        else:
            return eval_np(q, obs, acts)
        
    def get_pred_action_cem(self, q, obs, num_cem_iters=7, num_cem_samps=600, cem_frac=0.1, action_shape=7):
        #gauss_mean = np.zeros(4)
        gauss_mean = np.array([0]*action_shape)
        #gauss_std = np.ones(4)
        #gauss_std = np.diag(np.array([0.3, 0.3, 0.3, 0.3]))
        gauss_std = np.diag(np.ones(action_shape))
        for i in range(num_cem_iters):
            #acts = np.random.multivariate_normal(gauss_mean, np.diag(gauss_std**2), size=(num_cem_samps,))
            acts = np.random.multivariate_normal(gauss_mean, gauss_std, size=(num_cem_samps,))
            acts = acts.clip(min=[-1]*action_shape, max=[1]*action_shape)
            qvals = self.get_qval(q, np.repeat(obs.flatten()[None], acts.shape[0], 0), acts).squeeze()
            print('----- itr {} ----'.format(i))
            print('mean qval = {}'.format(qvals.mean()))
            print('std qval = {}'.format(qvals.std()))
            print('-----------------'.format(i))
            best_action_inds = (-qvals).argsort()[:int(num_cem_samps * cem_frac)]
            best_acts = acts[best_action_inds]
            gauss_mean = best_acts.mean(axis=0)
            #gauss_std = best_acts.std(axis=0)
            gauss_std = np.cov(best_acts, rowvar=False)
            print(gauss_std)
        print('cem choosing action', gauss_mean)
        print('q value of ', eval_np(q, obs[None], gauss_mean[None]))
        return gauss_mean


    def transform_images(self, images, is_image=False, single_viewpoint=True, show_image=False):
        #images here is shape (num_viewpoints, width,height, channels)
        if single_viewpoint:
            from PIL import Image
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt

            if not is_image:
                print(images.shape)
                images = images[0] # only want first viewpoint
                im = Image.fromarray(images)
            else:
                im = images

            if show_image:
                plt.figure()
                plt.imshow(np.asarray(im))
                plt.show()

            im = im.resize((64, 64), Image.ANTIALIAS)

            from torchvision import transforms
            import matplotlib.pyplot as plt
            trans = transforms.ToTensor()
            tens = trans(im).numpy().flatten()

            return tens
        else:
            assert False #TODO handle

    def act(self, t=None, i_tr=None, images=None, state=None, goal=None, goal_image=None, verbose_worker=None):
        self.policy = self.policy.to(ptu.device)
        self.qf1 = self.qf1.to(ptu.device)
        
        if t == 0 and self._hp.goal_cond:
            self.multiview_goal_imgs()
        if self._hp.log: 
            self.file_path = os.path.join(self.traj_log_dir, 'log.txt')

        if self._hp.goal_cond:
            print('error: ', np.array(self.goal_pos) - state[-1].squeeze()[:3])

        state = np.expand_dims(state[-1], axis=0)
        print(len(images))
        recent_img = images[-1]

        show_image = (t == 1)

        img_x = images[t-1]
        img_x = self.transform_images(img_x, show_image=show_image)

        if self._hp.optimize_q_function:
            action = self.get_pred_action_cem(self.qf1, img_x)
        else:
            action = self.policy.get_action(img_x.squeeze(), extra_fc_input=self.task_id_vec)[0]
        print(action.shape)

        if self._hp.log:
            file = open(self.file_path, 'a+')
            file.write(str(action) + '\n')
            #file.write(str(log_prob) + '\n')
            file.close()

        print('inferred action', action)
        
        if self._hp.normalize:
            action = (action + 1) / 2 * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW
        
        print('normalized action', action)
        
        output = AttrDict()
        output.actions = action
        
        return output
