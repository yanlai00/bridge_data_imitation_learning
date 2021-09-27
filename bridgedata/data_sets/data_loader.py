import torch.utils.data as data
import numpy as np
import glob
import h5py
import random
import imp
from bridgedata.data_sets.data_utils.test_datasets import make_gifs
from torch.utils.data import DataLoader
import os
from bridgedata.utils.general_utils import Configurable
from bridgedata.utils.general_utils import AttrDict, map_dict, resize_video
from bridgedata.data_sets.data_augmentation import get_random_color_aug, get_random_crop

class BaseVideoDataset(data.Dataset, Configurable):
    def __init__(self, data_conf, phase='train', shuffle=True):
        """

        :param data_dir:
        :param mpar:
        :param data_conf:
        :param phase:
        :param shuffle: whether to shuffle within batch, set to False for computing metrics
        :param dataset_size:
        """

        self._hp = self._default_hparams()
        self._override_defaults(data_conf)

        self.phase = phase
        self.data_conf = data_conf
        self.shuffle = shuffle and phase == 'train'

        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')

    def _default_hparams(self):
        default_dict = AttrDict(
            n_worker=10,
        )
        return AttrDict(default_dict)

    def get_data_loader(self, batch_size):
        print('datadir {}, len {} dataset {}'.format(self.data_conf.data_dir, self.phase, len(self)))
        print('data loader nworkers', self._hp.n_worker)
        return DataLoader(self, batch_size=batch_size, shuffle=self.shuffle, num_workers=self._hp.n_worker,
                                  drop_last=True)


class FixLenVideoDataset(BaseVideoDataset):
    """
    Variable length video dataset
    """

    def __init__(self, data_conf, phase='train', shuffle=True, transform=None):
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
        self.transform = transform

    def look_for_files(self, phase):
        if isinstance(self._hp.data_dir, list):
            self.filenames = []
            for dir in self._hp.data_dir:
                self.filenames += self._get_filenames(dir)
                random.seed(1)
                random.shuffle(self.filenames)
        else:
            self.filenames = self._get_filenames(self._hp.data_dir)
        self.filenames = self._maybe_post_split(self.filenames)
        if self._hp.train_data_fraction < 1 and phase == 'train':
            print('###################################')
            print("Warning, using {} fraction of data!!!".format(self._hp.train_data_fraction))
            print('###################################')
            self.filenames = self.filenames[:int(len(self.filenames) * self._hp.train_data_fraction)]

        if self._hp.max_train_examples and phase == 'train':
            print('###################################')
            print("Warning, using max train examples {}!!!".format(self._hp.max_train_examples))
            print('###################################')
            self.filenames = self.filenames[:self._hp.max_train_examples]

        self.traj_per_file = self.get_traj_per_file(self.filenames[0])
        if self._hp.T is None:
            self._hp.T = self.get_max_seqlen(self.filenames[0])
        print('init dataloader for phase {} with {} files'.format(phase, len(self.filenames)))

    def _default_hparams(self):
        # Data Dimensions
        default_dict = AttrDict(
            name="",   # the name of the dataset, used for writing logs
            data_dir=None,
            random_crop=False,
            image_size_beforecrop=None,
            color_augmentation=False,
            sel_len=-1,  # number of time steps for contigous sequence that is shifted within sequeence of T randomly
            sel_camera=0,
            concatentate_cameras=False,
            T=None,
            downsample_img_sz=None,
            train_data_fraction=1.,
            max_train_examples=None
        )
        # add new params to parent params
        parent_params = super()._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def _get_filenames(self, data_dir):
        assert 'hdf5' not in data_dir, "hdf5 most not be containted in the data dir!"
        filenames = sorted(glob.glob(os.path.join(data_dir, os.path.join('hdf5', self.phase) + '/*')))
        if not filenames:
            raise RuntimeError('No filenames found in {}'.format(data_dir))
        random.seed(1)
        random.shuffle(filenames)
        return filenames

    def get_traj_per_file(self, path):
        with h5py.File(path, 'r') as F:
            return int(np.array(F['traj_per_file']))

    def get_max_seqlen(self, path):
        # return maximum number of images over all trajectories
        with h5py.File(path, 'r') as F:
            return int(np.array(F['max_num_images']))

    def _get_num_from_str(self, s):
        return int(''.join(filter(str.isdigit, s)))

    def __getitem__(self, index):
        # making sure that different loading threads aren't using the same random seed.
        np.random.seed(index)
        random.seed(index)

        file_index = index // self.traj_per_file
        path = self.filenames[file_index]

        output = self.parse_file(path, index, self.traj_per_file)
        if self.transform is not None:
            return self.transform(output)
        else:
            return output

    def parse_file(self, path, index=0, traj_per_file=1):
        self.single_filename = str.split(path, '/')[-1]
        start_ind_str, _ = path.split('/')[-1][:-3].split('to')
        with h5py.File(path, 'r') as F:
            ex_index = index % traj_per_file  # get the index
            key = 'traj{}'.format(ex_index)
            data_dict = AttrDict()
            if key + '/images' in F:
                data_dict.images = np.array(F[key + '/images'])
            for name in F[key].keys():
                if name in ['states', 'actions', 'pad_mask']:
                    data_dict[name] = np.array(F[key + '/' + name]).astype(np.float32)
                if name in ['camera_ind', 'num_cameras', 'base_pos_ind', 'num_base_positions']: 
                    # camera index used in simulation when using a random virtual camera
                    data_dict[name] = np.array(F[key + '/' + name]).astype(np.int)


            if self._hp.T is not None:
                for key in data_dict.keys():
                    if key in ['camera_ind', 'num_cameras', 'base_pos_ind', 'num_base_positions', 'domain_ind']:
                        continue
                    if key == 'actions':  # actions are shorter by one time step
                        data_dict[key] = data_dict[key][:self._hp.T - 1]
                    else:
                        data_dict[key] = data_dict[key][:self._hp.T]

        data_dict = self.process_data_dict(data_dict)
        if self._hp.sel_len != -1:
            data_dict = self.sample_rand_shifts(data_dict)

        data_dict['tlen'] = data_dict['images'].shape[0]
        for k, v in data_dict.items():
            if k in ['camera_ind', 'num_cameras', 'base_pos_ind', 'num_base_positions', 'domain_ind', 'tlen']:
                continue
            if k == 'actions':
                desired_T = self._hp.T - 1 # actions need to be shorter by one since they need to have a start and end-state!
            else:
                desired_T = self._hp.T
            if v.shape[0] < desired_T:
                data_dict[k] = self.pad_tensor(v, desired_T)

        if 'camera_ind' in data_dict and 'base_pos_id' in data_dict:
            data_dict['domain_ind'] = data_dict['camera_ind'] * data_dict['num_base_positions'] + data_dict['base_pos_ind']
        elif 'camera_ind' in data_dict:
            data_dict['domain_ind'] = data_dict['camera_ind']
        return data_dict

    def process_data_dict(self, data_dict):
        if 'images' in data_dict:
            images = data_dict['images']
            if self._hp.sel_camera != -1:
                assert len(images.shape) == 5
                if self._hp.sel_camera == 'random':
                    cam_ind = np.random.randint(0, images.shape[1])
                    data_dict.camera_ind = np.array([cam_ind])
                    data_dict.num_cameras = images.shape[1]
                    images = images[:, cam_ind]
                else:
                    images = images[:, self._hp.sel_camera]
                    data_dict.camera_ind = np.array([self._hp.sel_camera])
                images = images[:, None]
            # Resize video
            if len(images.shape) == 5:
                imlist = []
                for n in range(images.shape[1]):
                    imlist.append(self.preprocess_images(images[:, n]))
                data_dict.images = np.stack(imlist, axis=1)
            else:
                data_dict.images = self.preprocess_images(images)
        return data_dict

    def sample_rand_shifts(self, data_dict):
        """ This function processes data tensors so as to have length equal to max_seq_len
        by sampling / padding if necessary """
        offset = np.random.randint(0, self.T - self._hp.sel_len, 1)

        data_dict = map_dict(lambda tensor: self._croplen(tensor, offset, self._hp.sel_len), data_dict)
        if 'actions' in data_dict:
            data_dict.action_targets = data_dict.action_targets[:-1]
        return data_dict

    def preprocess_images(self, images):
        assert images.dtype == np.uint8, 'image need to be uint8!'
        if self._hp.downsample_img_sz is not None:
            images = resize_video(images, (self._hp.downsample_img_sz[0], self._hp.downsample_img_sz[1]))
        images = np.transpose(images, [0, 3, 1, 2])  # convert to channel-first
        images = images.astype(np.float32) / 255
        if self._hp.color_augmentation and self.phase is 'train':
            images = get_random_color_aug(images, self._hp.color_augmentation)
        if self._hp.random_crop:
            assert images.shape[-2:] == tuple(self._hp.image_size_beforecrop)
            images = get_random_crop(images, self._hp.random_crop, center_crop=self.phase != 'train')
        images = images * 2 - 1
        assert images.dtype == np.float32, 'image need to be float32!'
        return images

    def pad_tensor(self, tensor, desired_T):
        pad = np.zeros([desired_T - tensor.shape[0]] + list(tensor.shape[1:]), dtype=np.float32)
        tensor = np.concatenate([tensor, pad], axis=0)
        return tensor

    def _maybe_post_split(self, filenames):
        """Splits dataset percentage-wise if respective field defined."""
        try:
            return self._split_with_percentage(self.data_conf.train_val_split, filenames)
        except (KeyError, AttributeError):
            return filenames

    def _split_with_percentage(self, frac, filenames):
        assert sum(frac.values()) <= 1.0  # fractions cannot sum up to more than 1
        assert self.phase in frac
        if self.phase == 'train':
            start, end = 0, frac['train']
        elif self.phase == 'val':
            start, end = frac['train'], frac['train'] + frac['val']
        else:
            start, end = frac['train'] + frac['val'], frac['train'] + frac['val'] + frac['test']
        start, end = int(len(filenames) * start), int(len(filenames) * end)
        return filenames[start:end]

    def __len__(self):
        return len(self.filenames) * self.traj_per_file

    @staticmethod
    def _croplen(val, offset, target_length):
        """Pads / crops sequence to desired length."""

        val = val[int(offset):]
        len = val.shape[0]
        if len > target_length:
            return val[:target_length]
        elif len < target_length:
            raise ValueError("not enough length")
        else:
            return val

    @staticmethod
    def get_dataset_spec(data_dir):
        return imp.load_source('dataset_spec', os.path.join(data_dir, 'dataset_spec.py')).dataset_spec



