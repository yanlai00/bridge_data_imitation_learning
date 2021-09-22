# from semiparametrictransfer.data_sets.robonet_dataloader import FilteredRoboNetDataset
from semiparametrictransfer.utils.general_utils import AttrDict
from visual_mpc.agent.utils.hdf5_saver import HDF5Saver
import argparse
import numpy as np
import os

def downsample_robonet():
    hp = AttrDict(
        name='robonet_sawyer',
        T=31,
        robot_list=['sawyer'],
        train_val_split=[1.0, 0, 0],
        # random_crop=True,
        # data_dir=os.environ['DATA'] + '/misc_datasets/robonet/robonet_sampler/hdf5',
        data_dir=os.environ['DATA'] + '/misc_datasets/robonet/robonet/hdf5',
        random_camera=False,
        img_sz=[56, 72]
    )
    output_folder = os.environ['DATA'] + '/misc_datasets/robonet/robonet/downsampled'

    batch_size = 10
    loader = FilteredRoboNetDataset(hp).get_data_loader(batch_size)
    agentparams = AttrDict(T=31)
    hdf5_saver = HDF5Saver(output_folder, None, agentparams, 1)
    for i_batch, sample_batched in enumerate(loader):
        images = np.asarray(sample_batched['images'])
        images = (np.transpose((images + 1) / 2, [0, 1, 2, 4, 5, 3]) * 255.).astype(np.uint8)  # convert to channel-first
        states = sample_batched['states']
        actions = np.asarray(sample_batched['actions'])
        # print('actions', actions)

        for b in range(batch_size):
            env_obs = {'images': images[b],
                       'state': states[b]}
            action_list = [{'actions': a_t} for a_t in actions[b]]
            hdf5_saver.save_traj(i_batch*batch_size + b, {}, obs=env_obs,
                                 policy_out=action_list)
        print('saved {} traj'.format(i_batch*batch_size + b))


if __name__ == '__main__':
    downsample_robonet()
