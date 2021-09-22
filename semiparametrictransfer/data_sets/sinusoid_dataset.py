import torch.utils.data as data
import numpy as np
from torch.utils.data import DataLoader
from semiparametrictransfer.utils.general_utils import AttrDict
import torch
import _pickle as pkl
from torch.autograd import Variable


def create_sinusoid_dataset():
    n_data_persinus = 100
    n_sinus = 100

    # additive noise
    std_dev = 0.01

    x_range = [-1, 1]

    l_sinus_samples = []
    for i in range(n_sinus):
        # sample phases
        pi = np.pi
        phase = np.random.uniform(0, 2 * pi)
        scale = np.random.uniform(1, 2)
        bias = np.random.uniform(-1, 1)
        x_scale = np.random.uniform(1, 4)

        x_vals = np.random.uniform(x_range[0], x_range[1], n_data_persinus)

        y_vals = [scale * np.sin(x * x_scale + phase) + bias + np.random.normal(0, std_dev) for x in x_vals]
        l_sinus_samples.append(AttrDict(x_vals=x_vals, y_vals=y_vals))

    def stack_and_shuffle(l_sinus):
        x_vals = np.concatenate([sinus.x_vals for sinus in l_sinus])
        y_vals = np.concatenate([sinus.y_vals for sinus in l_sinus])

        data = np.stack([x_vals, y_vals], axis=1)
        np.random.shuffle(data)
        return data

    train_split = int(n_sinus * 0.9)
    train_data = stack_and_shuffle(l_sinus_samples[:train_split])
    val_data = stack_and_shuffle(l_sinus_samples[train_split:])

    return AttrDict(l_sinus_samples=l_sinus_samples,
                    train=train_data,
                    val=val_data,
                    )


def create_piecewise_sinusoid_dataset(save_path=None):
    n_data_persinus = 2000
    n_sinus = 1

    # additive noise
    std_dev = 0.01

    x_range = [-5, 5]

    l_sinus_samples = []

    x_start_end = np.sort(np.random.uniform(x_range[0], x_range[1], n_sinus * 2))

    for i in range(n_sinus):
        # sample phases
        pi = np.pi
        phase = np.random.uniform(0, 2 * pi)
        scale = np.random.uniform(0.1, 1)
        bias = np.random.uniform(-0.1, 0.1)
        x_scale = np.random.uniform(1, 2)

        x_vals = np.random.uniform(x_start_end[i * 2], x_start_end[i * 2 + 1], n_data_persinus)

        y_vals = [scale * np.sin(x * x_scale + phase) + bias + np.random.normal(0, std_dev) for x in x_vals]
        l_sinus_samples.append(AttrDict(x_vals=x_vals, y_vals=y_vals))

    def stack_and_shuffle(l_sinus):
        x_vals = np.concatenate([sinus.x_vals for sinus in l_sinus])
        y_vals = np.concatenate([sinus.y_vals for sinus in l_sinus])

        data = np.stack([x_vals, y_vals], axis=1)
        np.random.shuffle(data)
        return data

    train_split = int(np.ceil(n_sinus * 0.9))
    np.random.shuffle(l_sinus_samples)
    train_val_data = stack_and_shuffle(l_sinus_samples[:train_split])
    # held_out_sinus = stack_and_shuffle(l_sinus_samples[train_split:])

    train_split = int(train_val_data.shape[0]*0.95)
    train_data = train_val_data[:train_split]
    val_data = train_val_data[train_split:]

    data = AttrDict(l_sinus_samples=l_sinus_samples,
                    train=train_data,
                    val=val_data,
                    # held_out_sinus=held_out_sinus
                    )

    if save_path is not None:
        pkl.dump(data, open(save_path, 'wb'))

    return data

class SinusoidDataset(data.Dataset):
    def __init__(self, phase, n_worker=10, embedding_size=32, shuffle=True):
        data_path = '/nfs/kun1/users/febert/data/semiparametrictransfer/trainingdata/single_sinus.pkl'
        self.n_worker = n_worker
        data = torch.from_numpy(pkl.load(open(data_path, "rb"))[phase])

        import pdb; pdb.set_trace()
        self.data_x = data[:, 0].float()
        self.data_y = data[:, 1].float()

        self.embedding_vector = torch.zeros([self.data_x.shape[0], embedding_size], dtype=torch.float32)
        self.shuffle = shuffle
        self.phase = phase

    def get_data_loader(self, batch_size):
        print('len {} dataset {}'.format(self.phase, len(self)))
        return DataLoader(self, batch_size=batch_size, shuffle=self.shuffle, num_workers=self.n_worker,
                                  drop_last=True)

    def set_embedding(self, embedding, index):
        self.embedding_vector[index] = embedding

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, index):
        return AttrDict(data_x=self.data_x[index], data_y=self.data_y[index], index=index)
