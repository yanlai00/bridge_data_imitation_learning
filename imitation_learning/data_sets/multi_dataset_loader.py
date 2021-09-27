import torch.utils.data as data
from semiparametrictransfer.utils.general_utils import AttrDict
import time
import hashlib
import numpy as np
from torch.utils.data import DataLoader
import os

class MultiDatasetLoader():
    def __init__(self, dataset_dict, phase, shuffle=True):
        """
        :param dataset_dict:  Attrdict with {single_task: AttrDict(dataset_class=..., data_conf=AttrDict()),}
        :param phase:
        :param shuffle:
        """
        self.data_sets = AttrDict()
        self.lengths = AttrDict()
        self._hp = AttrDict()

        if shuffle:
            self.n_worker = 10
        else:
            self.n_worker = 1
        if 'n_worker' in dataset_dict:
            self.n_worker = dataset_dict.pop('n_worker')
        for dataset_name, value in dataset_dict.items():
            dataset_class = value.dataclass
            data_conf = value.dataconf
            self.data_sets[dataset_name] = dataset_class(data_conf, phase, shuffle)
            self.lengths[dataset_name] = len(self.data_sets[dataset_name])
            self._hp[dataset_name] = data_conf

        self.phase = phase
        self.shuffle = shuffle and phase == 'train'

    def __getitem__(self, index):
        dict = AttrDict()
        for name, dataset in self.data_sets.items():
            use_index = index % self.lengths[name]
            dict[name] = dataset.__getitem__(use_index)
        return dict

    def __len__(self):
        lengths = [len(d) for d in self.data_sets.values()]
        return max(lengths)

    def get_data_loader(self, batch_size):
        lengths = [len(d) for d in self.data_sets.values()]
        print('phase {} len {} nworker:{} '.format(self.phase, lengths, self.n_worker))
        return DataLoader(self, batch_size=batch_size, shuffle=self.shuffle, num_workers=self.n_worker,
                          drop_last=True)


class RandomMixingDatasetLoader():
    def __init__(self, dataset_dict, phase, shuffle):
        """
        :param dataset_dict:  Attrdict with {dataset_name: (dataset_class, data_conf)}
        :param phase:
        :param shuffle:
        """
        self.data_sets = {}
        self.lengths = {}
        self._hp = dataset_dict
        self.data_set_sample_probabilities = []
        for key, value  in dataset_dict.items():
            if key.startswith('dataset'):
                dataset_name, [dataset_class, data_conf, prob] = key, value
                self.data_set_sample_probabilities.append(prob)
                self.data_sets[dataset_name] = dataset_class(data_conf, phase, shuffle)
                self.lengths[dataset_name] = len(self.data_sets[dataset_name])
        
        self.sync_train_domain_and_taskdescription_indices()

        self.phase = phase
        self.data_conf = data_conf
        self.shuffle = shuffle and phase == 'train'

        if shuffle:
            self.n_worker = 10
        else:
            self.n_worker = 1

        # self.n_worker = 0
        self._hp.name = 'random_mixing'

    def __getitem__(self, index):
        """
        :param index:  index referes to the index of the shortest datasets datapoints for the other datasets are selected randomly
        :return:
        """
        np.random.seed(index)
        name = str(np.random.choice(list(self.data_sets.keys()), 1, p=self.data_set_sample_probabilities)[0])
        use_index = index % self.lengths[name]
        # print('index {} useindex {} length {}'.format(index, use_index, self.lengths[name]))
        dict = self.data_sets[name].__getitem__(use_index)
        return dict

    def __len__(self):
        lengths = [len(d) for d in self.data_sets.values()]
        return max(lengths)

    def get_data_loader(self, batch_size):
        print('len {} dataset {}'.format(self.phase, len(self)))
        return DataLoader(self, batch_size=batch_size, shuffle=self.shuffle, num_workers=self.n_worker,
                          drop_last=True)

    def sync_train_domain_and_taskdescription_indices(self):
        dataset_names = list(self.data_sets.keys())
        all_domains = set([d for dataset_name in dataset_names for d in list(self.data_sets[dataset_name].domain_hash_index.keys())])
        all_taskdescriptions = set([d for dataset_name in dataset_names for d in list(self.data_sets[dataset_name].taskdescription2task_index.keys())])

        self.domain_hash_index = {domain_hash: index for domain_hash, index in
                                               zip(all_domains, range(len(all_domains)))}
        self.taskdescription2task_index = {task_descp: index for task_descp, index in
                                               zip(all_taskdescriptions, range(len(all_taskdescriptions)))}
        print('taskdescription2task_index', self.taskdescription2task_index)
    
    def set_domain_and_taskdescription_indices(self, domain_index, task_index):
        """
        This is to make sure that the train and val dataloaders are using the same domain_has_index and taskdescription2task_index
        """
        for dataset in list(self.data_sets.values()):
            dataset.domain_hash_index = domain_index
            dataset.taskdescription2task_index = task_index
    
    @property
    def dataset_stats(self):
        return "\n".join([dataset.dataset_stats for dataset in list(self.data_sets.values())])



def count_hashes(loader):
    tstart = time.time()
    single_task_hashes = set()
    bridge_data_hashes = set()
    n_batch_counter = 0
    for i_batch, sample_batched in enumerate(loader):
        # print('ibatch', counter)
        add_hashes(single_task_hashes, sample_batched['single_task']['images'])
        add_hashes(bridge_data_hashes, sample_batched['bridge_data']['images'])
        n_batch_counter += 1
        if n_batch_counter % 500 == 0:
            print('batch_counter', n_batch_counter)

    print('batch_counter', n_batch_counter)
    print('num hashes single task', len(single_task_hashes))
    print('num hashes bridge task', len(bridge_data_hashes))
    print('average loading time', (time.time() - tstart) / n_batch_counter)


def add_hashes(hashes, images):
    for b in range(images.shape[0]):
        image_string = images[b].numpy().tostring()
        hashes.add(hashlib.sha256(image_string).hexdigest())

