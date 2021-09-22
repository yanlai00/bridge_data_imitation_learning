from semiparametrictransfer.policies.bc_policy import BCPolicyStates
import numpy as np
import pickle as pkl
from semiparametrictransfer.utils.general_utils import AttrDict


class SPTPolicy(BCPolicyStates):
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        super().__init__(ag_params, policyparams, gpu_id, ngpu)
        # load dataset of state trajecotries
        self.data_dict = pkl.load(open(self._hp.aux_data_dir + '/gtruth_nn_train.pkl', "rb"))

    def _default_hparams(self):
        default_dict = {
            'aux_data_dir': None
        }

        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])

        parent_params.model = SPTModelTest
        return parent_params

    def get_nearest_neighbors(self, goal_state_sequences):

        def get_displacements(all_obj_qpos ):
            obj_displacements = all_obj_qpos[:, -1] - all_obj_qpos[:, 0]
            obj_displacements_mag = np.linalg.norm(obj_displacements, axis=-1)
            largest_displacement_index = np.argmax(obj_displacements_mag, axis=1)
            # get largest obj displacement per trajectory
            largest_displacement = np.stack(
                [obj_displacements[i, ind] for i, ind in enumerate(largest_displacement_index)])
            return obj_displacements, largest_displacement, largest_displacement_index

        n_objects = 3
        origin_object_qpos = goal_state_sequences[:, 9:15].reshape(goal_state_sequences.shape[0], n_objects, 2)
        _, origin_largest_displacement, orig_lrgst_disp_ind = get_displacements(origin_object_qpos[None])

        target_object_qpos = np.stack([self.data_dict[key]['object_qpos'] for key in self.data_dict.keys()])
        target_displacment, _, _ = get_displacements(target_object_qpos)
        target_data_dict_keys = [k for k in self.data_dict.keys()]

        # compute the magnitude of differences between i-th displacement vector and all other displacements
        diff_mag = np.linalg.norm(
            target_displacment[:, orig_lrgst_disp_ind] - origin_largest_displacement[None], axis=-1)

        # get the batch indices of the lowest dist:
        numbest_k = 10
        best_ind = np.argsort(diff_mag.squeeze())[:numbest_k]
        print('bestind {} largest disp {}, best 3 diffmag: {}'.format( best_ind[:10],
                                                                       origin_largest_displacement,
                                                                       np.sort(diff_mag)[:3]))
        nearest_traj = [target_data_dict_keys[i] for i in best_ind]

        nn_states = []
        nn_actions = []
        nn_images = []
        for traj in nearest_traj:
            data = read_traj(self._hp.aux_data_dir + '/hdf5/train/' + traj)
            nn_states.append(data['states'])
            nn_actions.append(data['actions'])
            nn_images.append(data['images'])
        nn_states = np.stack(nn_states, axis=1)[None]
        nn_actions = np.stack(nn_actions, axis=1)[None]
        nn_images = np.stack(nn_images, axis=1)[None]

        return nn_states, nn_actions, nn_images

    def act(self, t=None, i_tr=None, images=None, state=None, loaded_traj_info=None, goal_image=None):
        nn_states, nn_actions, nn_images = self.get_nearest_neighbors(loaded_traj_info['state'])

        self.t = t
        self.i_tr = i_tr
        goal_states = loaded_traj_info['state'][-1]

        inputs = AttrDict(sel_states=self.npy2trch(state[-1][None]),
                          goal_states=self.npy2trch(goal_states[None]),
                          best_matches_states=self.npy2trch(nn_states),
                          best_matches_actions=self.npy2trch(nn_actions))

        out = self.predictor(inputs)

        output = AttrDict()
        output.actions = out['a_pred'].data.cpu().numpy()[0]

        if self._hp.verbose:
            if t == 0:
                vid = assemble_videos_kbestmatches(self.npy2trch(images[-1]),
                                                   self.npy2trch((goal_image[-1]*255).astype(np.uint8)),
                                                   self.npy2trch(nn_images.squeeze()[None]),
                                                   n_batch_examples=1)
                vid = [np.transpose(frame, [1,2,0]).astype(np.uint8) for frame in vid]
                npy_to_gif(vid, self.traj_log_dir + '/nn_vis_t{}'.format(t))
        return output


