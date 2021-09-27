import numpy as np
import torch
import random
from tqdm import tqdm
import copy
from bridgedata.utils.general_utils import AttrDict
# from bridgedata.data_sets.robonet_dataloader import FilteredRoboNetDataset
from bridgedata.utils.general_utils import Configurable
from bridgedata.data_sets.data_augmentation import get_random_color_aug, get_random_crop
from bridgedata.utils.general_utils import select_indices
import os



def make_data_aug(images, hp, phase):
    batch_size = images.shape[0]
    if hp.color_augmentation and phase is 'train':
        aug_images = []
        for b in range(batch_size):
            aug_images.append(get_random_color_aug(images[b][None], hp.color_augmentation, minus_one_to_one_range=True))
        aug_images = np.concatenate(aug_images, 0)
        images = aug_images* 2 - 1
    if hp.random_crop:
        assert images.shape[-2:] == tuple(hp.image_size_beforecrop)
        cropped_images = []
        for b in range(batch_size):
            cropped_images.append(get_random_crop(images[b][None], hp.random_crop, center_crop= phase != 'train'))
        images = np.concatenate(cropped_images, 0)
    if not isinstance(images, torch.Tensor):
        images = torch.from_numpy(images).float()
    return images

def apply_data_aug(dict, hp, phase):
    for key, value in dict.items():
        if 'image' in key:
            dict[key] = make_data_aug(value, hp, phase)
    return dict

class DatasetReplayBuffer(Configurable):
    def __init__(self, data_conf, phase='train', shuffle=True):
        self._hp = self._default_hparams()
        self._override_defaults(data_conf)
        self.phase = phase

        print('making replay buffer for', data_conf.data_dir)
        load_data_conf = copy.deepcopy(data_conf)
        dataset_class = load_data_conf.pop('dataset_type')
        # drop keys that are not used in data loader
        if 'max_datapoints' in load_data_conf:
            load_data_conf.pop('max_datapoints')

        # remove data augmentation since we want to have non-augmented data in the replay buffer
        if 'color_augmentation' in load_data_conf:
            load_data_conf.pop('color_augmentation')
        if 'random_crop' in load_data_conf:
            load_data_conf.pop('random_crop')
        if 'sel_camera' in load_data_conf:
            if load_data_conf['sel_camera'] == 'random':
                load_data_conf.pop('sel_camera')
        if 'debug' in load_data_conf:
            load_data_conf.pop('debug')
        if 'num_cams_per_variation' in load_data_conf:
            load_data_conf.pop('num_cams_per_variation')
        self.loader = dataset_class(load_data_conf, phase).get_data_loader(1)
        self.buffer = []
        self.loadDataset()
        self.get_data_counter = 0

    def _default_hparams(self):
        # Data Dimensions
        default_dict = AttrDict(
            name="",  # the name of the dataset, used for writing logs
            dataset_type=None,
            max_train_examples=None,
            data_dir=None,
            n_worker=10,
            random_crop=False,
            image_size_beforecrop=None,
            color_augmentation=False,
            sel_len=-1,  # number of time steps for contigous sequence that is shifted within sequeence of T randomly
            sel_camera=None,
            num_cams_per_variation=None,
            concatentate_cameras=False,
            T=None,
            downsample_img_sz=None,
            train_data_fraction=1.,
            robot_list=None,
            camera=0,
            target_adim=None,
            target_sdim=None,
            splits="",
            debug=False
        )
        return AttrDict(default_dict)

    def random_batch(self):
        indices = np.random.randint(0, len(self.buffer), self.batch_size)
        output_dict = AttrDict()
        selected_dicts = np.array(self.buffer)[indices]
        for key in selected_dicts[0]:
            output_dict[key] = torch.cat([sel[key] for sel in selected_dicts])

        t0 = np.array([np.random.randint(0, tlen - 1) for tlen in output_dict['tlen']])
        output_dict['final_image'] = select_indices(output_dict.images, output_dict['tlen'] - 1).squeeze()
        for tag in ['states', 'actions', 'images']:
            output_dict[tag] = select_indices(output_dict[tag], t0).squeeze()

        if 'sel_camera' in self._hp:
            if self._hp.sel_camera == 'random':
                ncam = output_dict.images.shape[1]
                cam_ind = np.random.randint(0, ncam, self.batch_size)
                if self._hp.num_cams_per_variation is not None:
                    camera_variation_index = output_dict['camera_variation_index']
                    output_dict['domain_ind'] = (camera_variation_index * self._hp.num_cams_per_variation + cam_ind).to(torch.long)
                else:
                    output_dict['domain_ind'] = cam_ind
                output_dict.images = select_indices(output_dict.images, cam_ind).squeeze()
                output_dict['final_image'] = select_indices(output_dict['final_image'], cam_ind).squeeze()
        apply_data_aug(output_dict, self._hp, self.phase)
        return output_dict

    def loadDataset(self):
        print('loading dataset into replay buffer...')
        for sampled_batch in tqdm(self.loader):
            self.buffer.append(sampled_batch)
            # if self._hp.max_datapoints is not None:
            #     if len(self.buffer) > self._hp.max_datapoints:
            #         print('max data points reached!')
            #         break

            if self._hp.debug:
                if len(self.buffer) > 10:
                    # import pdb; pdb.set_trace()
                    print('break at 10!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11')
                    break
        print('done loading.')

    def __next__(self):
        self.get_data_counter += 1
        if self.get_data_counter < self.__len__():
            batch = self.random_batch()
            return batch
        else:
            raise StopIteration

    def __iter__(self):
        self.get_data_counter = 0
        return self

    def get_data_loader(self, batch_size):
        self.batch_size = batch_size
        return self

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def __len__(self):
        if self.batch_size is None:
            raise NotImplementedError('length only implemented for loader!')
        return int(len(self.buffer * 50)/self.batch_size) # iterate through the same data 50 times.

class MultiDatasetReplayBuffer():
    def __init__(self, dataset_dict, phase, shuffle=True):
        self.data_sets = AttrDict()
        self._hp = AttrDict()
        for dataset_name, value in dataset_dict.items():
            data_conf = value.dataconf
            self.data_sets[dataset_name] = DatasetReplayBuffer(data_conf, phase, shuffle)
            self._hp[dataset_name] = data_conf

        self.phase = phase

    def __next__(self):
        self.get_data_counter += 1
        if self.get_data_counter < self.__len__():
            dict = AttrDict()
            for name, dataset in self.data_sets.items():
                dict[name] = dataset.random_batch()
            return dict
        else:
            raise StopIteration

    def __iter__(self):
        self.get_data_counter = 0
        return self

    def __len__(self):
        lengths = [len(d) for d in self.data_sets.values()]
        return min(lengths)

    def get_data_loader(self, batch_size):
        for dataset in self.data_sets.values():
            dataset.set_batch_size(batch_size)
        return self
