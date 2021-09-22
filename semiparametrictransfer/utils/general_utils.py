import numpy as np
import re
import cv2
from PIL import Image
from torchvision.transforms import Resize
import torch
from functools import partial, reduce
import copy

def str2int(str):
    try:
        return int(str)
    except ValueError:
        return None


class Configurable:
    def _override_defaults(self, params):
        params = copy.copy(params)
        if 'identical_default_ok' in params:
            identical_default_ok = True
            params.pop('identical_default_ok')
        else:
            identical_default_ok = False

        for name, value in params.items():
            # print('overriding param {} to value {}'.format(name, value))
            if value == getattr(self._hp, name) and not identical_default_ok:
                raise ValueError("attribute is {} is identical to default value {} !!".format(name, value))
            self._hp[name] = value

    def _default_hparams(self):
        return AttrDict()

class HasParameters:
    def __init__(self, **kwargs):
        self.build_params(kwargs)

    def build_params(self, inputs):
        # If params undefined define params
        try:
            self.params
        except AttributeError:
            self.params = self.get_default_params()
            self.params.update(inputs)

    # TODO allow to access parameters by self.<param>


def move_to_device(inputs, device):
    def func(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(x, list):
            return list(map(lambda x:x.to(device), x))
        if isinstance(x, dict):
            return AttrDict(map_dict(func, x))
        else:
            return x.to(device)
    return AttrDict(map_dict(func, inputs))


def down_sample_imgs(obs, des_size):
    obs = copy.deepcopy(obs)
    imgs = obs['images']
    target_array = np.zeros([imgs.shape[0], imgs.shape[1], des_size[0], des_size[1], 3], dtype=np.uint8)
    for n in range(imgs.shape[1]):
        for t in range(imgs.shape[0]):
            target_array[t, n] = cv2.resize(imgs[t, n], (des_size[1], des_size[0]), interpolation=cv2.INTER_AREA)
    obs['images'] = target_array
    return obs

class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self): return self
    def __setstate__(self, d): self = d


def np_unstack(array, axis):
    arr = np.split(array, array.shape[axis], axis)
    arr = [a.squeeze() for a in arr]
    return arr

def map_dict(fn, d):
    """takes a dictionary and applies the function to every element"""
    return type(d)(map(lambda kv: (kv[0], fn(kv[1])), d.items()))


def make_recursive(fn, *argv, **kwargs):
    """ Takes a fn and returns a function that can apply fn on tensor structure
     which can be a single tensor, tuple or a list. """

    def recursive_map(tensors):
        if tensors is None:
            return tensors
        elif isinstance(tensors, list) or isinstance(tensors, tuple):
            return type(tensors)(map(recursive_map, tensors))
        elif isinstance(tensors, dict):
            return type(tensors)(map_dict(recursive_map, tensors))
        elif isinstance(tensors, torch.Tensor):
            return fn(tensors, *argv, **kwargs)
        else:
            try:
                return fn(tensors, *argv, **kwargs)
            except Exception as e:
                print("The following error was raised when recursively applying a function:")
                print(e)
                raise ValueError("Type {} not supported for recursive map".format(type(tensors)))

    return recursive_map


def listdict2dictlist(LD):
    """ Converts a list of dicts to a dict of lists """

    # Take intersection of keys
    keys = reduce(lambda x, y: x & y, (map(lambda d: d.keys(), LD)))
    return AttrDict({k: [dic[k] for dic in LD] for k in keys})

def make_recursive_list(fn):
    """ Takes a fn and returns a function that can apply fn across tuples of tensor structures,
     each of which can be a single tensor, tuple or a list. """

    def recursive_map(tensors):
        if tensors is None:
            return tensors
        elif isinstance(tensors[0], list) or isinstance(tensors[0], tuple):
            return type(tensors[0])(map(recursive_map, zip(*tensors)))
        elif isinstance(tensors[0], dict):
            return map_dict(recursive_map, listdict2dictlist(tensors))
        elif isinstance(tensors[0], torch.Tensor):
            return fn(*tensors)
        else:
            try:
                return fn(*tensors)
            except Exception as e:
                print("The following error was raised when recursively applying a function:")
                print(e)
                raise ValueError("Type {} not supported for recursive map".format(type(tensors)))

    return recursive_map


recursively = make_recursive


def map_recursive(fn, tensors):
    return make_recursive(fn)(tensors)


def map_recursive_list(fn, tensors):
    return make_recursive_list(fn)(tensors)

def resize_video(video, size):
    transformed_video = np.stack([np.asarray(Resize(size)(Image.fromarray(im))) for im in video], axis=0)
    return transformed_video

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def select_indices(tensor, indices):
    assert len(indices.shape) == 1
    new_images = []
    for b in range(tensor.shape[0]):
        new_images.append(tensor[b, indices[b]])
    tensor = torch.stack(new_images, dim=0)
    return tensor

class RecursiveAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0

    def update(self, val):
        self.val = val
        if self.sum is None:
            self.sum = val
        else:
            self.sum = map_recursive_list(lambda x, y: x + y, [self.sum, val])
        self.count += 1
        self.avg = map_recursive(lambda x: x / self.count, self.sum)


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def trch2npy(tensor):
    return tensor.data.cpu().numpy()

def npy2trch(tensor, device='cuda'):
    return torch.from_numpy(tensor).to(torch.device(device))
