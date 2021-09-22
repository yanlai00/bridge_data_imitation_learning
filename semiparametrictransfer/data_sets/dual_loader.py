import torch.utils.data as data
import numpy as np
from collections import OrderedDict
import torch
from PIL import Image
import glob
import h5py
import pickle as pkl
import random
import pdb
import matplotlib.pyplot as plt
import imageio
import imp
from torch.utils.data import DataLoader
import os
from semiparametrictransfer.utils.general_utils import Configurable
from semiparametrictransfer.utils.general_utils import AttrDict, map_dict, resize_video
from semiparametrictransfer.data_sets.data_augmentation import get_random_color_aug, get_random_crop
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset

class DualVideoDataset(FixLenVideoDataset, Configurable):
    def __init__(self, data_conf, phase='train', shuffle=True):
        """
        :param data_conf:  Attrdict with keys
        :param phase:
        :param shuffle: whether to shuffle within batch, set to False for computing metrics
        :param dataset_size:
        """
        super().__init__(data_conf, phase, shuffle)
        self._hp = self._default_hparams()
        self._override_defaults(data_conf)
        self.look_for_files(phase)

    def look_for_files(self, phase):
        self.filenames = AttrDict()
        self.traj_per_file = AttrDict()
        for name, dir in self._hp.data_dir.items():
            self.filenames[name] = self._maybe_post_split(self._get_filenames(dir))
            self.traj_per_file[name] = self.get_traj_per_file(self.filenames[name][0])
            if self._hp.T is None:
                self._hp.T = AttrDict()
                self._hp.T[name] = self.get_total_seqlen(self.filenames[name][0])
            print('init dataloader {} for phase {} with {} files'.format(name, phase, len(self.filenames[name])))
        self._hp.camera = 'all'

    def __getitem__(self, index):
        dict = AttrDict()
        for name, dir in self._hp.data_dir.items():
            if self.lengths[name] == np.min(np.array(list(self.lengths.values()))):
                use_index = index
            else:
                use_index = int(np.random.randint(0, self.lengths[name], 1))
            file_index = use_index // self.traj_per_file[name]
            path = self.filenames[name][file_index]
            dict[name] = self.parse_file(path, use_index, self.traj_per_file[name])
        return dict

    def __len__(self):
        self.lengths = {k: len(v)*self.traj_per_file[k] for k, v in self.filenames.items()}
        return np.min(np.array(list(self.lengths.values())))
