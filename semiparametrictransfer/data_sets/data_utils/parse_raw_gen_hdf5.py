import argparse
import glob
import copy
from visual_mpc.agent.utils.hdf5_saver import HDF5Saver
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.data_sets.data_utils.clone_smalldata_for_loading_efficiency import clone_data
import json
import random
import sys
import os
import shutil
import math
import pickle as pkl
import pdb;
import cv2
import numpy as np
from semiparametrictransfer.utils.general_utils import down_sample_imgs

from semiparametrictransfer.data_sets.data_utils.compute_normalization import compute_dataset_normalization
from semiparametrictransfer.utils.general_utils import map_dict

class StageLengthException(Exception):
    pass


def get_folder_names(root_dir):
    dirs = glob.glob(root_dir + '/*/raw/*/*')
    print('found {} folders.'.format(len(dirs)))
    return dirs

def get_traj_start_ind(stage):
    for i in range(len(stage)):
        if stage[i] == 1:
            break
    return i

def parse_traj(cameras, folder, num_stages=1, max_num_states=None):
    env_obs = pkl.load(open('{}/obs_dict.pkl'.format(folder), 'rb'), encoding='latin1')
    if 'states' in env_obs:
        env_obs['state'] = env_obs['states']

    if len(cameras) == 0:
        n_cams = len(glob.glob('{}/images*'.format(folder)))
        cam_ind = range(n_cams)
    else:
        cam_ind = cameras
        n_cams = len(cameras)

    if n_cams:
        T = min([len(glob.glob('{}/images{}/*.png'.format(folder, i))) for i in range(n_cams)])
        height, width = cv2.imread('{}/images0/im_0.png'.format(folder)).shape[:2]
        env_obs['images'] = np.zeros((T, n_cams, height, width, 3), dtype=np.uint8)

        for n in cam_ind:
            for time in range(T):
                env_obs['images'][time, n] = cv2.imread('{}/images{}/im_{}.png'.format(folder, n, time))[:, :, ::-1]

    policy_out = pkl.load(open('{}/policy_out.pkl'.format(folder), 'rb'), encoding='latin1')

    if num_stages == 1:
        return [env_obs, policy_out, T]
    else:
        ind = get_traj_start_ind(env_obs['task_stage'])
        if ind == 0:
            print("stage length is zero! skpping this one.")
            raise StageLengthException
        if ind > max_num_states:
            print("stage length is too big! skpping this one.")
            raise StageLengthException
        print('start stage 1 index', ind)
        out = [[map_dict(lambda x: x[:ind], env_obs), policy_out[:ind - 1], None],
               [map_dict(lambda x: x[ind:], env_obs), policy_out[ind: T -1], None]]
        return out


def extract_and_save_hdf5(folders, output_folder, target_size, split, max_num_states, cameras, num_stages=1):
    if max_num_states == -1:
        [_, _, max_num_states] = parse_traj(cameras, folders[0])
    print('num images per traj: ', max_num_states)
    agentparams = AttrDict(max_num_actions=max_num_states - 1)

    if num_stages == 1:
        check_exists(output_folder)
        hdf5_savers = [HDF5Saver(output_folder, None, agentparams, 1, split=split)]
        output_folders = [output_folder]
    else:
        output_folders = [output_folder + '/stage{}'.format(i) for i in range(num_stages)]
        for folder in output_folders:
            check_exists(folder)
        hdf5_savers = [HDF5Saver(folder, None, agentparams, 1, split=split) for folder in output_folders]

    for i, traj in enumerate(folders):
        print('saving traj', i)
        try:
            trajdata_list = parse_traj(traj, num_stages, max_num_states)
        except StageLengthException:
            continue

        for trajdata, saver in zip(trajdata_list, hdf5_savers):
            env_obs, policy_out, _ = trajdata
            saver.save_traj(i, {}, obs=down_sample_imgs(env_obs, target_size),
                                      policy_out=policy_out)
    return output_folders


def check_exists(output_folder):
    if os.path.exists(output_folder + '/hdf5'):
        print('path {} exisits'.format(output_folder + '/hdf5'))
        response = input('do you want to delete it y/n?')
        if response == 'y':
            shutil.rmtree(output_folder + '/hdf5')
        else:
            raise ValueError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="converts dataset from pkl format to hdf5")
    parser.add_argument('input_folder', type=str, help='where raw files are stored')
    parser.add_argument('--output_folder', default='', type=str, help='where to save')
    parser.add_argument('--prefix', default='', type=str, help='prefix added to save directory')
    parser.add_argument('--Tmax', default=-1, type=int, help='maximum number of time-steps, finds Tmax automatically for fixed len datasets')
    parser.add_argument('--nstages', default=1, type=int, help='specify number of stages')
    parser.add_argument('--large', action='store_true', help='double image size')
    parser.add_argument("--cameras", type=int, nargs='+', default=[],
                        help="list of cameras to include")
    args = parser.parse_args()

    folders = get_folder_names(args.input_folder)
    if args.output_folder is '':
        args.output_folder = copy.deepcopy(args.input_folder)
    if args.prefix is not '':
        args.output_folder += '/' + args.prefix
    print('saving to ', args.output_folder)

    if not args.large:
        target_size = np.array([48, 64]) + np.array([8, 8])
    else:
        target_size = np.array([96, 128]) + np.array([16, 16])
    split = (0.90, 0.10, 0.0)
    print('using image size ', target_size)

    output_folders = extract_and_save_hdf5(folders, args.output_folder, target_size, split, args.Tmax, args.nstages)
    for folder in output_folders:
        args = AttrDict(data_dir=folder)
        compute_dataset_normalization(folder)

    # clone data for better loading speed
    if len(folders) < 1000:
        for folder in output_folders:
            clone_data(folder)
