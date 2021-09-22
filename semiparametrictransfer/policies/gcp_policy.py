import numpy as np
import torch
from visual_mpc.policy.policy import Policy
from semiparametrictransfer.models.gcbc import GCBCModelTest
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.models.gcp import GCPModelTest
from semiparametrictransfer.utils.general_utils import np_unstack

class GCPPolicyStates(Policy):
    """
conf_invembed.py    Behavioral Cloning Policy
    """
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        super(GCPPolicyStates, self).__init__()

        self._hp = self._default_hparams()
        self._override_defaults(policyparams)

        update_dict = {
        'batch_size' : 1,
        'state_dim': ag_params['state_dim'],
        'action_dim': ag_params['action_dim']
        }
        self.T = ag_params['T']
        self.pred_len = self.T - 1

        self._hp.gcbc_params.update(update_dict)
        self._hp.gcp_params.update(update_dict)

        self.gcbc_predictor = self._hp.gcbc_model(self._hp.gcbc_params)
        self.gcbc_predictor.eval()
        self.gcp_predictor = self._hp.gcp_model(self._hp.gcp_params)
        self.gcp_predictor.eval()

        self.device = torch.device('cpu')

        self.current_plan = None
        self.all_plans = []
        self.tplan = 0

    def reset(self):
        super().reset()

    def _default_hparams(self):
        default_dict = {
            'gcbc_params': {},
            'gcp_params': {},
            'gcbc_model': GCBCModelTest,
            'gcp_model': GCPModelTest,
            'verbose': False,
            'replan_interval': 10
        }

        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def act(self, image=None, t=None, i_tr=None, state=None, loaded_traj_info=None):
        self.t = t
        self.i_tr = i_tr
        goal_state = loaded_traj_info['state'][-1]
        goal_state = self.npy2trch(goal_state[None])

        if t % self._hp.replan_interval == 0:
            self.tplan = 0
            inputs = AttrDict(state=self.npy2trch(state[-1][None]),
                              goal_state=goal_state)
            self.current_plan = self.gcp_predictor(inputs, self.pred_len)
            self.all_plans.append(self.current_plan)

        out = self.gcbc_predictor(self.current_plan[self.tplan], goal_state)
        self.tplan += 1
        output = AttrDict()
        output.actions = out['a_pred'].data.cpu().numpy()[0]

        if self._hp.verbose and self.t == self.T-1:
            self.visualize_plan(image)
        return output

    def visualize_plan(self, image):

        im_height = image.shape[1]
        im_width = image.shape[2]
        total_width = (self.T * 2 + 1) * im_width
        total_height = len(self.all_plans)*im_height

        out = np.zeros((total_height, total_width, 3))

        out[:im_height, :] = np.concatenate(np_unstack(image, axis=0), 1)

        for p, plan in enumerate(self.all_plans):

            import pdb; pdb.set_trace()
            colstart = p*self._hp.replan_interval*im_width
            out[(p+1)*im_height: (p+2)*im_height, colstart: colstart + self.pred_len * im_width] = plan


    @property
    def default_action(self):
        return np.zeros(self.predictor._hp.n_actions)

    def log_outputs_stateful(self, logger=None, global_step=None, phase=None, dump_dir=None, exec_seq=None, goal=None, index=None, env=None, goal_pos=None, traj=None, topdown_image=None):
        logger.log_video(np.transpose(exec_seq, [0, 3, 1, 2]), 'control/traj{}_'.format(index), global_step, phase)
        goal_img = np.transpose(goal, [2, 0, 1])[None]
        goal_img = torch.tensor(goal_img)
        logger.log_images(goal_img, 'control/traj{}_goal'.format(index), global_step, phase)

    def npy2trch(self, arr):
        return torch.from_numpy(arr).float().to(self.device)




