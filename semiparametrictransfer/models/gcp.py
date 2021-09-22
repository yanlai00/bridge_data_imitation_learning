import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from semiparametrictransfer.utils.general_utils import AttrDict, map_dict
from semiparametrictransfer.utils.general_utils import select_indices
from semiparametrictransfer.models.base_model import BaseModel
from semiparametrictransfer.models.utils.recurrent import LSTM

from semiparametrictransfer.utils.logger import Mujoco_Renderer


class GCPModel(BaseModel):
    """Semi parametric transfer model"""
    def __init__(self, overrideparams, logger=None):
        super().__init__(logger)
        self._hp = self._default_hparams()
        self.overrideparams = overrideparams
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
            random_tshifts=False,
            autoregressive=False
        )
        # add new params to parent params
        parent_params = super()._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def build_network(self):
        self.lstm = LSTM(self._hp.state_dim*2, self._hp.state_dim, 128, 3, self._hp.batch_size)

    def forward(self, inputs, tlen=None):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """

        tlen = inputs.states.shape[1]
        states = inputs.states
        goal_state = states[:, -1]
        return self.rollout_rnn(states, goal_state, tlen)

    def rollout_rnn(self, states, goal_state, tlen):
        self.lstm.init_hidden()
        if self._hp.apply_dataset_normalization:
            states = self.apply_dataset_normalization(states, 'states')
        # todo normalize goal_state as well!
        pred_states = []
        for t in range(tlen - 1):
            ncontext = 1
            if self._hp.autoregressive:
                if t < ncontext:
                    current_state = states[:, t]
                else:
                    current_state = pred_states[-1]
                pred_states.append(self.lstm(torch.cat([current_state, goal_state], dim=-1)))
            else:
                # teacher forcing
                pred_states.append(self.lstm(torch.cat([states[:, t], goal_state], dim=-1)))
        pred_states = torch.stack(pred_states, dim=1)
        outdict = AttrDict()
        if self._hp.apply_dataset_normalization:
            outdict.un_norm_pred_states = pred_states
            outdict.pred_states = self.unnormalize_dataset(pred_states, 'states')
        else:
            outdict.pred_states = pred_states
        return outdict

    def loss(self, model_input, model_output):
        gtruth_states = model_input.states
        if self._hp.apply_dataset_normalization:
            gtruth_states = self.apply_dataset_normalization(gtruth_states, 'states')
            pred_states = model_output.un_norm_pred_states
        else:
            pred_states = model_output.pred_states
        losses = AttrDict(mse=torch.nn.MSELoss()(pred_states, gtruth_states[:, 1:]))

        # compute total loss
        losses.total_loss = torch.stack(list(losses.values())).sum()
        return losses

    def _log_outputs(self, model_output, inputs, losses, step, phase):
        video_freq = 200
        if step % video_freq:
            pred_traj = self.render_examples(model_output.pred_states)
            gtruth_traj = self.render_examples(inputs.states[:, 1:])

            vid = torch.cat([gtruth_traj, pred_traj], dim=2)
            vid = torch.cat(torch.unbind(vid, dim=0), dim=2).permute(0, 3, 1, 2)
            self._logger.log_video(vid, 'pred', step, phase, fps=10)

    def render_examples(self, states):
        n_ex = 5
        mj = Mujoco_Renderer(48, 64)
        videos = []
        for n in range(n_ex):
            vid = []
            for t in range(states.shape[1]):
                frame = mj.render(states[n, t].data.cpu().numpy()[:15])
                vid.append(torch.from_numpy(frame))
            videos.append(torch.stack(vid, dim=0))
        videos = torch.stack(videos, dim=0)
        return videos


class GCPModelTest(GCPModel):
    def __init__(self, overridparams, logger=None):
        super(GCPModelTest, self).__init__(overridparams, logger=logger)
        self._restore_params()
        self._hp.autoregressive = True

    def forward(self, inputs, tlen=None):
        return self.rollout_rnn(inputs.state[None], inputs.goal_state, tlen)
