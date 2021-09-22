import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.utils.general_utils import select_indices
from semiparametrictransfer.models.base_model import BaseModel
from semiparametrictransfer.models.utils.layers import BaseProcessingNet

class TrajFollowModel(BaseModel):
    """Semi parametric transfer model"""
    def __init__(self, overrideparams, logger=None):
        super().__init__(logger)
        self._hp = self._default_hparams()
        self.overrideparams = overrideparams
        self._override_defaults(overrideparams)  # override defaults with config file
        assert self._hp.batch_size != -1
        assert self._hp.state_dim != -1
        assert self._hp.action_dim != -1
        self.actions = None
        self.T = self._hp.T
        self.build_network()

    def _default_hparams(self):
        default_dict = AttrDict(
            state_dim=-1,
            action_dim=-1,
            traj_embed_size=64,
            encode_actions=False,
            T=30
        )
        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def sample_tsteps(self, states, actions):
        self.sel_times = np.random.randint(0, self.T, self._hp.batch_size)

        # there is no action corresponding to state
        sel_states = select_indices(states, self.sel_times)
        actions = select_indices(actions, self.sel_times)
        return sel_states, actions

    def build_network(self):
        # initialize trajectory encoder
        if self._hp.encode_actions:
            self.state_enc_ndim = self._hp.state_dim + self._hp.action_dim + 1
        else:
            self.state_enc_ndim = self._hp.state_dim + 1

        self.single_state_encoder = BaseProcessingNet(self.state_enc_ndim, 128,
                                                      self._hp.traj_embed_size, num_layers=2)
        self.traj_encoder_head = BaseProcessingNet(self._hp.traj_embed_size, 128,
                                                  self._hp.traj_embed_size, num_layers=2)
        self.action_decoder = BaseProcessingNet(self._hp.traj_embed_size + self._hp.state_dim, 128,
                                                  self._hp.action_dim, num_layers=3)

    def get_nn(self, inputs):
        return inputs.best_matches_states, inputs.best_matches_actions

    def embed_traj(self, actions, states):
        """

        :param best_matches:  [b, t, kbest, state_dim]
        :return:
        """
        time_input = torch.arange(self.T)[None, :, None].repeat(self.bsize, 1,  1).float().to(self._hp.device)
        if self._hp.encode_actions:
            enc_input = torch.cat([actions[:, :self.T], states[:, :self.T], time_input], dim=-1)
        else:
            enc_input = torch.cat([states[:, :self.T], time_input], dim=-1)
        enc = self.single_state_encoder(enc_input.view(self.bsize * self.T, self.state_enc_ndim))
        enc = torch.mean(enc.view(self.bsize, self.T, self._hp.traj_embed_size), dim=1)
        return self.traj_encoder_head(enc)

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
        self.bsize = inputs.states.shape[0]
        self.sel_states, self.sel_actions = self.sample_tsteps(inputs.states, inputs.action_targets)

        embed = self.embed_traj(inputs.action_targets, inputs.states)
        a_pred = self.action_decoder(torch.cat([embed, self.sel_states], dim=-1))
        return AttrDict(a_pred=a_pred)

    def loss(self, model_input, model_output):
        losses = AttrDict()
        losses['mse'] = torch.nn.MSELoss()(model_output.a_pred, self.sel_actions)
        # compute total loss
        losses.total_loss = torch.stack(list(losses.values())).sum()
        return losses



class TrajFollowModelTest(TrajFollowModel):
    def __init__(self, overridparams, logger=None):
        super(TrajFollowModelTest, self).__init__(overridparams, logger)
        self._restore_params()

    def forward(self, inputs):
        self.bsize = 1
        embed = self.embed_traj(None, inputs.reference_statetraj)
        a_pred = self.action_decoder(torch.cat([embed, inputs.state], dim=-1))
        return AttrDict(a_pred=a_pred)


