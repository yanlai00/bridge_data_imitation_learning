
import os
import glob
import h5py
import numpy as np


def read_traj(path):
    with h5py.File(path, 'r') as F:
        ex_index = 0
        key = 'traj{}'.format(ex_index)

        data_dict = {}
        # Fetch data into a dict
        for name in F[key].keys():
            if name in ['states', 'actions', 'pad_mask']:
                data_dict[name] = F[key + '/' + name].value.astype(np.float32)
    return data_dict

def save_traj(path, filename, data_dict):
    if not os.path.exists(path):
        os.makedirs(path)
    out_filename = os.path.join(path, filename)
    with h5py.File(out_filename, 'w') as F:
        key = 'traj0'
        for name, value in data_dict.items():
            F[key + '/' + name] = value
        F['traj_per_file'] = 1
        print('writing ', out_filename)

def _get_filenames(data_dir, phase):
    assert 'hdf5' not in data_dir, "hdf5 most not be containted in the data dir!"
    filenames = sorted(glob.glob(os.path.join(data_dir, os.path.join('hdf5', phase) + '/*')))
    if not filenames:
        raise RuntimeError('No filenames found in {}'.format(data_dir))
    return filenames

def convert(orig_path, dest_path):
    dest_path += '/hdf5'
    os.makedirs(dest_path, exist_ok=True)
    phases = ['train', 'val', 'test']
    for phase in phases:
        filenames = _get_filenames(orig_path, phase)
        print('found {} traj for {}'.format(len(filenames), phase))
        for path in filenames:
            single_filename = str.split(path, '/')[-1]
            data_dict = read_traj(path)
            save_traj(os.path.join(dest_path, phase), single_filename, data_dict)


if __name__ == '__main__':
    convert(os.environ['DATA'] + '/spt_trainingdata' + '/sim/tabletop-texture', os.environ['DATA'] + '/spt_trainingdata' + '/sim/tabletop-texture-statesonly')


