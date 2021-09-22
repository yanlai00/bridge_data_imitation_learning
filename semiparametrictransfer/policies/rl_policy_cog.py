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
# from rlkit.torch.conv_networks import CNN, ConcatCNN, ConcatBottleneckCNN, TwoHeadCNN
from rlkit.torch.conv_networks import CNN, ConcatCNN, ConcatBottleneckCNN, TwoHeadCNN,  VQVAEEncoderConcatCNN, \
    ConcatBottleneckVQVAECNN, VQVAEEncoderCNN, MultiToweredCNN
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic

import h5py
import pickle

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import cv2


from semiparametrictransfer.utils.general_utils import AttrDict


class RLPolicyCOG(Policy):
    def __init__(self, ag_params, policyparams, gpu_id=0):
        self._adim = ag_params['env'][0].adim
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)

        self.enable_gpus(str(gpu_id))
        ptu.set_gpu_mode(True)
        parameters = torch.load(self._hp.path)
        action_dim = 7

        if self._hp.history:
            self.prev_img = [np.zeros((1, self._hp.flattened_image_size)) for i in range(self._hp.history_size)]
        
        cnn_params=dict(
            kernel_sizes=[3, 3, 3],
            n_channels=[16, 16, 16],
            strides=[1, 1, 1],
            hidden_sizes=[1024, 512, 256],
            paddings=[1, 1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2, 1],  # the one at the end means no pool
            pool_strides=[2, 2, 1],
            pool_paddings=[0, 0, 0],
            image_augmentation=True,
            image_augmentation_padding=4,
            spectral_norm_conv=False,
            spectral_norm_fc=False,
        )

        cnn_params.update(
            input_width=64,
            input_height=64,
            input_channels=9 if self._hp.history else 3,
            output_size=1,
            added_fc_input_size=action_dim,
            normalize_conv_activation=False
        )
        cnn_params.update(
            output_size=256,
            added_fc_input_size=0,
            hidden_sizes=[1024, 512],
            spectral_norm_fc=False,
            spectral_norm_conv=False,
            normalize_conv_activation=False
        )

        if self._hp.vqvae:
            policy_obs_processor = VQVAEEncoderCNN(**cnn_params, num_res=3)
        else:
            policy_obs_processor = CNN(**cnn_params)

        self.policy = TanhGaussianPolicy(
            obs_dim=cnn_params['output_size'],
            action_dim=action_dim,
            hidden_sizes=[256, 256, 256],
            obs_processor=policy_obs_processor,
        )

        self.policy.load_state_dict(parameters['policy_state_dict'])

        self.load_qfunc = False
        if self.load_qfunc:
            if self._hp.bottleneck:
                self.qf1 = ConcatBottleneckCNN(action_dim, bottleneck_dim=16,deterministic=False, width=64, height=64)
            if self._hp.vqvae:
                cnn_params = dict(
                    kernel_sizes=[3, 3, 3],
                    n_channels=[16, 16, 16],
                    strides=[1, 1, 1],
                    hidden_sizes=[1024, 512, 256],
                    paddings=[1, 1, 1],
                    pool_type='max2d',
                    pool_sizes=[2, 2, 1],
                    pool_strides=[2, 2, 1],
                    pool_paddings=[0, 0, 0],
                    image_augmentation=True,
                    image_augmentation_padding=4,
                )
                cnn_params.update(
                    input_width=64,
                    input_height=64,
                    input_channels=9 if self._hp.history else 3,
                    output_size=1,
                    added_fc_input_size=action_dim,
                )
                self.qf1 = VQVAEEncoderConcatCNN(**cnn_params, num_res = 3)
            else:
                cnn_params=dict(
                        kernel_sizes=[3, 3, 3],
                        n_channels=[16, 16, 16],
                        strides=[1, 1, 1],
                        hidden_sizes=[1024, 512, 256],
                        paddings=[1, 1, 1],
                        pool_type='max2d',
                        pool_sizes=[2, 2, 1],
                        pool_strides=[2, 2, 1],
                        pool_paddings=[0, 0, 0],
                        image_augmentation=True,
                        image_augmentation_padding=4,
                )
                cnn_params.update(
                        input_width=64,
                        input_height=64,
                        input_channels=9 if self._hp.history else 3,
                        output_size=1,
                        added_fc_input_size=action_dim,
                )
                self.qf1 = ConcatCNN(**cnn_params)
            self.qf1.load_state_dict(parameters['qf1_state_dict'])

    def enable_gpus(self, gpu_str):
        if gpu_str is not "":
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

    def _default_hparams(self):
        default_dict = {
            'user': False,
            'path': '/home/asap7772/batch_rl_private/data/lagrange-10-robonet-widowx/302109/lagrange_10_robonet_widowx/302109_2020_05_27_01_42_24_0000--s-0',
            'policy_type': 1,
            'goal_pos': (0.935024, 0.204873, 0.0694792),
            'data_aug': False,
            'data_aug_version': 0,
            'goals_path': '/raid/asap7772/3cam_widowx_data.hdf5',
            'log': False,
            'num_views': 1,
            'goal_cond': False,
            'goal_cond_version': 'gc_img',
            'optimize_q_function': False,
            'bottleneck': False,
            'vqvae': False,
            'history':False,
            'history_size': 3,
            'flattened_image_size': 3*64*64,
        }
        parent_params = super(RLPolicyCOG, self)._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def set_log_dir(self, d):
        print('setting log dir')
        super(RLPolicyCOG, self).set_log_dir(d)

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


    def transform_images(self, images, is_image=False, single_viewpoint=True):
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


            show_image = True
            if show_image:
                plt.figure()
                plt.imshow(np.asarray(im))
                plt.show()

            width, height = im.size  # Get dimensions
            new_width, new_height = 480, 480
            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height) / 2
            # Crop the center of the image
            im = im.crop((left, top, right, bottom))
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
        if self.load_qfunc:
            self.qf1 = self.qf1.to(ptu.device)
        
        if t == 0 and self._hp.goal_cond:
            self.multiview_goal_imgs()

        if self._hp.log: self.file_path = os.path.join(self.traj_log_dir, 'log.txt')

        if self._hp.user:
            import ipdb
            ipdb.set_trace()

        if self._hp.goal_cond:
            print('error: ', np.array(self.goal_pos) - state[-1].squeeze()[:3])

        state = np.expand_dims(state[-1], axis=0)
        print(len(images))
        recent_img = images[-1]

        img_x = self.transform_images(images[t])

        if self._hp.history:
            self.prev_img.append(img_x)
            self.prev_img.pop(0)
            self.prev_img = [x.astype(img_x.dtype).reshape(-1,3,64,64) for x in self.prev_img]
            img_x = np.concatenate(tuple(self.prev_img),axis=1).reshape(1,-1)

        if self._hp.optimize_q_function:
            action = self.get_pred_action_cem(self.qf1, img_x)
        else:
            action = self.policy.get_action(img_x.squeeze())[0]
        
        print(action.shape)

        debug = False
        if debug:
            obs = torch.from_numpy(img_x.squeeze()).to(ptu.device)

            def show_heatmaps(obs, pred_action = None):
                import matplotlib.pyplot as plt
                x = np.linspace(-0.8,0.8)
                y = np.flip(np.linspace(-0.8,0.8))
                actions = torch.from_numpy(np.array(np.meshgrid(x,y)))
                actions = actions.flatten(1).T


                if pred_action is None:
                    actions_close = torch.cat((actions, torch.zeros((actions.shape[0],5))), axis=1).float().to(ptu.device)
                    actions_open = torch.cat((actions, torch.zeros((actions.shape[0],4)), torch.ones_like(actions)[:, :1]), axis=1).float().to(ptu.device)
                else:
                    actions_close = pred_action[None].repeat(actions.shape[0],0)
                    actions_open = pred_action[None].repeat(actions.shape[0],0)

                    actions_close[:,:2] = actions
                    actions_open[:, :2] = actions

                    actions_open[:,-1] = 0
                    actions_close[:, -1] = 1

                    actions_open = ptu.from_numpy(actions_open)
                    actions_close = ptu.from_numpy(actions_close)


                obs_tens = obs.repeat(actions.shape[0], 1, 1, 1).flatten(1).to(ptu.device)
                # qf1 = lambda x, y: y.sum(axis=1, keepdim=True)

                columns=np.around(x,decimals=2)
                index=np.around(y,decimals=2)
                columns = [str(x) for x in columns]
                index = [str(x) for x in index]

                self.policy = self.policy.cuda()
                qvals = self.policy.log_prob(obs_tens, actions_close)
                # qvals[qvals <= 0] = 0

                qvals = qvals.detach().cpu().numpy().flatten().reshape(50,50)
                df = pd.DataFrame(qvals, columns=columns, index=index)
                ax = sns.heatmap(df)
                plt.show()

                qvals = self.policy.log_prob(obs_tens, actions_open, extra_fc_input = None)

                # qvals[qvals <= 0] = 0

                qvals = qvals.detach().cpu().numpy().flatten().reshape(50,50)
                df = pd.DataFrame(qvals, columns=columns, index=index)
                ax = sns.heatmap(df)
                plt.show()

            #current obs heatmap
            # show_heatmaps(obs, pred_action=action)

            # import ipdb; ipdb.set_trace()

            show_dataset_heatmaps = False
            if show_dataset_heatmaps:
                from PIL import Image

                dataset_image = Image.open('/home/datacol1/trainingdata/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/put_potato_on_plate/2021-07-05_15-33-28/raw/traj_group0/traj0/images0/im_0.jpg')
                actions = pickle.load(open('/home/datacol1/trainingdata/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/put_potato_on_plate/2021-07-05_15-33-28/raw/traj_group0/traj0/policy_out.pkl','rb'))

                obs_data = ptu.from_numpy(self.transform_images(dataset_image,is_image=True))
                show_heatmaps(obs_data)


        if self._hp.log:
            file = open(self.file_path, 'a+')
            file.write(str(action) + '\n')
            #file.write(str(log_prob) + '\n')
            file.close()

        print('inferred action', action)
        
        normalize = True
        if normalize:
            action[:6] /= 20
            action[6] = 1 - action[6]
        
        print('normalized action', action)
        
        output = AttrDict()
        output.actions = action
        
        
        return output
