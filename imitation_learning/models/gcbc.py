import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from imitation_learning.utils.general_utils import AttrDict
from imitation_learning.utils.general_utils import select_indices
from imitation_learning.models.base_model import BaseModel
from imitation_learning.models.utils.layers import BaseProcessingNet

class GCBCModel(BaseModel):
    """Semi parametric transfer model"""
    def __init__(self, overrideparams, logger=None):
        super().__init__(overrideparams, logger)
        self._hp = self._default_hparams()
        self._override_defaults(overrideparams)  # override defaults with config file

        assert self._hp.batch_size != -1
        assert self._hp.state_dim != -1
        assert self._hp.action_dim != -1

        self.build_network()
        self.actions = None

    def _default_hparams(self):
        default_dict = AttrDict(
            state_dim=-1,
            action_dim=-1,
            goal_cond=True,
            goal_state_delta_t=None,
            use_conv=False
        )
        # add new params to parent params
        parent_params = super()._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def sample_tsteps(self, states, actions):
        tlen = states.shape[1]

        # get positives:
        t0 = np.random.randint(0, tlen-1, self._hp.batch_size)

        sel_states = select_indices(states, t0)
        if self._hp.goal_cond:
            if self._hp.goal_state_delta_t is not None:
                tg = t0 + np.random.randint(1, self._hp.goal_state_delta_t + 1, self._hp.batch_size)
                tg = np.clip(tg, 0, tlen - 1)
                goal_states = select_indices(states, tg)
            else:
                goal_states = states[:, -1]
            action_pred_input = torch.cat([sel_states, goal_states], dim=1)
        else:
            action_pred_input = sel_states
        actions = select_indices(actions, t0)
        return action_pred_input, actions

    def build_network(self):
        if self._hp.goal_cond:
            inputdim = self._hp.state_dim*2
        else:
            inputdim = self._hp.state_dim
        self.s_encoder = BaseProcessingNet(inputdim, 128,
                                           self._hp.action_dim, num_layers=3, normalization=None)

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
        action_pred_input, self.actions = self.sample_tsteps(inputs.states, inputs.action_targets)
        a_pred = self.s_encoder.forward(action_pred_input)
        return AttrDict(a_pred=a_pred)

    def loss(self, model_input, model_output):
        losses = AttrDict(mse=torch.nn.MSELoss()(model_output.a_pred, self.actions))

        # compute total loss
        losses.total_loss = torch.stack(list(losses.values())).sum()
        return losses




class GCBCModelTest(GCBCModel):
    def __init__(self, overridparams, logger=None):
        super(GCBCModelTest, self).__init__(overridparams, logger)
        self._restore_params()


    def forward(self, inputs):
        if self._hp.goal_cond:
            a_pred_input = torch.cat([inputs.state, inputs.goal_state], dim=1)
        else:
            a_pred_input = inputs.state
        a_pred = self.s_encoder.forward(a_pred_input)
        return AttrDict(a_pred=a_pred)


