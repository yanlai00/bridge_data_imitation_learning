from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
import os
import _pickle as pkl
from copy import deepcopy
import matplotlib.pyplot as plt
from semiparametrictransfer.utils.general_utils import AttrDict
import numpy as np
import pdb
from tqdm import tqdm
import h5py
from collections import OrderedDict
from semiparametrictransfer.utils.construct_html import save_gif_list_direct
from semiparametrictransfer.utils.construct_html import fill_template, save_html_direct
import os
import glob


def read_traj(path):
    with h5py.File(path, 'r') as F:
        ex_index = 0
        key = 'traj{}'.format(ex_index)

        data_dict = {}
        # Fetch data into a dict
        for name in F[key].keys():
            if name in ['states', 'actions']:
                data_dict[name] = F[key + '/' + name].value.astype(np.float32)
        states = data_dict['states']
        n_objects = 3
        data_dict['object_qpos'] = states[:, 9:15].reshape(states.shape[0], n_objects, 2)
        data_dict['images'] = F[key + '/images'].value
    return data_dict


def _get_filenames(data_dir, phase):
    assert 'hdf5' not in data_dir, "hdf5 most not be containted in the data dir!"
    filenames = sorted(glob.glob(os.path.join(data_dir, os.path.join('hdf5', phase) + '/*')))
    if not filenames:
        raise RuntimeError('No filenames found in {}'.format(data_dir))
    return filenames

def load_phase_data(orig_path, phase):
    all_data_dict = OrderedDict()
    # filenames = _get_filenames(orig_path, phase)[:100]  ###################
    filenames = _get_filenames(orig_path, phase)
    print('found {} traj for {}'.format(len(filenames), phase))
    print('loading files')
    for path in tqdm(filenames):
        single_filename = str.split(path, '/')[-1]
        all_data_dict[single_filename] = read_traj(path)
    return all_data_dict


def compute_nearest_neighbors(origin_data_dict, target_data_dict):
    """
    :param origin_data_dict:  data dict with data point for which we want to find neighbors in target_data_dict
    :param target_data_dict:
    :return:
    """
    nearest_ind = {}
    def get_displacements(data_dict):
        all_obj_qpos = np.stack([data_dict[key]['object_qpos'] for key in data_dict.keys()])
        obj_displacements = all_obj_qpos[:, -1] - all_obj_qpos[:, 0]
        obj_displacements_mag = np.linalg.norm(obj_displacements, axis=-1)
        largest_displacement_index = np.argmax(obj_displacements_mag, axis=1)
        # get largest obj displacement per trajectory
        largest_displacement = np.stack([obj_displacements[i, ind] for i, ind in enumerate(largest_displacement_index)])
        return obj_displacements, largest_displacement, largest_displacement_index

    origin_displacment, origin_largest_displacement, orig_lrgst_disp_ind = get_displacements(origin_data_dict)
    target_displacment, _, _ = get_displacements(target_data_dict)
    target_data_dict_keys = [k for k in target_data_dict.keys()]

    for i, k in tqdm(enumerate(origin_data_dict.keys())):
        # compute the magnitude of differences between i-th displacement vector and all other displacements
        diff_mag = np.linalg.norm(target_displacment[:, orig_lrgst_disp_ind[i]] - origin_largest_displacement[i][None], axis=-1)
        # get the batch indices of the lowest dist:
        numbest_k = 128
        best_ind = np.argsort(diff_mag)[:numbest_k]
        print('i {}: bestind {} largest disp {}, best 3 diffmag: {}'.format(i, best_ind[:10], origin_largest_displacement[i], np.sort(diff_mag)[:3]))
        nearest_ind[i] = best_ind
        origin_data_dict[k]['nearest_ind'] = [target_data_dict_keys[i] for i in best_ind]

    #### debug:
    # first_three_keys = [k for k in origin_data_dict.keys()][:3]
    # print('nearest of first three keys')
    # for i in range(3):
    #     print('according to dict')
    #     print(origin_data_dict[first_three_keys[i]]['nearest_ind'][:5])
    #     print('according to nearest_ind list')
    #     print(nearest_ind[i][:5])
    # import pdb; pdb.set_trace()
    return origin_data_dict, nearest_ind


def save_nearest_neighbors(data_dict, phase, base_path):
    all_data_dict_ = deepcopy(data_dict)
    all_data_dict_noimages = {}
    for key in data_dict.keys():
        all_data_dict_[key].pop('images')
        all_data_dict_noimages[key] = all_data_dict_[key]
    pkl.dump(all_data_dict_noimages,
             open(base_path + '/gtruth_nn_{}.pkl'.format(phase), 'wb'))

def save_html(nearest_ind, show_num_gifs, html_paths, save_path):
    itemdict = {}
    # show only the nearest neighbors for the top 10
    for i in range(show_num_gifs):
        nearest_ind_i = nearest_ind[i][:show_num_gifs]
        nearest_paths = [html_paths[ind] for ind in nearest_ind_i]
        #         import pdb; pdb.set_trace()
        itemdict['img{}'.format(i)] = [html_paths[i]] + nearest_paths

    html_page = fill_template(itemdict, exp_name='Nearest Neighbors')
    save_html_direct(save_path + '/visuals/index.html', html_page)


def save_gifs(nearest_ind, show_num_gifs, data_dict, base_path):
    images = np.stack([data_dict[key]['images'] for key in data_dict.keys()]).squeeze()
    all_inds = []

    for k in range(show_num_gifs):
        all_inds.append(k)
        all_inds.extend(nearest_ind[k][:show_num_gifs])

    all_inds = set(all_inds)
    print('saving {} traj'.format(len(all_inds)))
    gif_list = [(ind, images[ind]) for ind in all_inds]

    folder = base_path + '/visuals'
    name = 'sawyer'
    html_paths = save_gif_list_direct(folder, name, gif_list)
    return html_paths

def create_nearest_neighbors(base_path):
    train_data_dict = load_phase_data(base_path, phase='train')
    train_data_dict, train_nearest_ind = compute_nearest_neighbors(train_data_dict, train_data_dict)
    html_paths = save_gifs(train_nearest_ind, 20, train_data_dict, base_path)
    save_html(train_nearest_ind, 20, html_paths, base_path)
    save_nearest_neighbors(train_data_dict, 'train', base_path)

    val_data_dict = load_phase_data(base_path, phase='val')
    val_data_dict, val_nearest_ind = compute_nearest_neighbors(val_data_dict, train_data_dict)

    save_nearest_neighbors(val_data_dict, 'val', base_path)



if __name__ == '__main__':
    base_path = os.environ['DATA'] + '/spt_trainingdata' + '/sim/tabletop-texture'
    create_nearest_neighbors(base_path)



