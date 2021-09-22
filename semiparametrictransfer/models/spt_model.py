import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.utils.general_utils import select_indices
from semiparametrictransfer.models.base_model import BaseModel
from semiparametrictransfer.models.utils.layers import BaseProcessingNet


class SPTModel(BaseModel):
    """Semi parametric transfer model"""
    def __init__(self, overrideparams, logger=None):
        super().__init__(logger)
        self._hp = self._default_hparams()
        self._override_defaults(overrideparams)  # override defaults with config file
        assert self._hp.batch_size != -1
        assert self._hp.state_dim != -1
        assert self._hp.action_dim != -1

        if hasattr(self._hp.data_conf, 'n_best'):
            self._hp.n_best = self._hp.data_conf.n_best
        self.build_network()
        self.actions = None

    def _default_hparams(self):
        default_dict = AttrDict(
            state_dim=-1,
            action_dim=-1,
            traj_embed_size=64,
            goal_cond=True,
            n_best=10,
            separate_pred_gains=False   # calculate prediction gains separately for each conditioning traj
        )
        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def sample_image_pair(self, inputs):
        self.sel_times = np.random.randint(0, self.T, self._hp.batch_size)
        inputs.current_img = select_indices(inputs.images, self.sel_times).squeeze()
        inputs.sel_states = select_indices(inputs.states, self.sel_times)
        inputs.sel_actions = select_indices(inputs.action_targets, self.sel_times)
        return inputs

    def build_network(self):
        if self._hp.goal_cond:
            action_pred_input_ndim = self._hp.state_dim * 2 + self._hp.traj_embed_size
        else:
            action_pred_input_ndim = self._hp.state_dim + self._hp.traj_embed_size
        self.action_predictor = BaseProcessingNet(action_pred_input_ndim, 128,
                                                  self._hp.action_dim, num_layers=3)
        # action predictor that does not get auxiliary inputs:
        self.action_predictor_no_aux = BaseProcessingNet(action_pred_input_ndim, 128,
                                                  self._hp.action_dim, num_layers=3)
        # initialize trajectory encoder
        self.single_state_encoder = BaseProcessingNet(self._hp.state_dim + self._hp.action_dim + 1, 128,
                                                      self._hp.traj_embed_size, num_layers=3)
        self.traj_encoder_head = BaseProcessingNet(self._hp.traj_embed_size, 128,
                                                  self._hp.traj_embed_size, num_layers=2)

    def get_nn(self, inputs):
        return inputs.best_matches_states, inputs.best_matches_actions

    def embed_traj(self, best_matches_states, best_matches_actions):
        """

        :param best_matches:  [b, t, kbest, state_dim]
        :return:
        """
        time_input = torch.arange(self.T)[None, :, None, None].repeat(self.bsize, 1, self._hp.n_best, 1).float().to(self._hp.device)
        best_matches = torch.cat([best_matches_states[:, :self.T, :self._hp.n_best],
                                  best_matches_actions[:, :self.T, :self._hp.n_best], time_input], dim=-1)
        enc = self.single_state_encoder(best_matches.view(self.bsize*self._hp.n_best*self.T, self._hp.state_dim + self._hp.action_dim + 1))
        enc = torch.mean(enc.view(self.bsize, self._hp.n_best*self.T, self._hp.traj_embed_size), dim=1)
        return self.traj_encoder_head(enc)

    def predict_actions(self, action_pred_input, best_match_embed):
        return self.action_predictor.forward(torch.cat([action_pred_input,
                                                          best_match_embed], dim=-1))

    def encode_sel_timestep(self, inputs):
        pass

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
        self.T = inputs.action_targets.shape[1]
        self.bsize = inputs.states.shape[0]
        inputs = self.sample_image_pair(inputs)
        inputs.goal_states = inputs.states[:, -1]
        return self.forward_pass(inputs)

    def forward_pass(self, inputs):
        if self._hp.goal_cond:
            action_pred_input = torch.cat([inputs.sel_states, inputs.goal_states], dim=1)
        else:
            action_pred_input = inputs.sel_states

        # encode query point
        enc_selected_t = self.encode_sel_timestep(action_pred_input)

        # retrieve best matches
        best_matches_states, best_matches_actions = self.get_nn(inputs, enc_selected_t)
        # embed best matches:
        best_match_embed = self.embed_traj(best_matches_states, best_matches_actions)
        # predict actions w/ auxiliary input
        a_pred = self.predict_actions(action_pred_input, best_match_embed)
        # predict actions w/o auxiliary input
        a_pred_no_aux = self.action_predictor_no_aux.forward(torch.cat([action_pred_input,
                                                                        torch.zeros_like(best_match_embed)], dim=-1))
        return AttrDict(a_pred=a_pred, a_pred_no_aux=a_pred_no_aux)

    def loss(self, model_input, model_output):
        losses = AttrDict()
        losses['mse'] = torch.nn.MSELoss()(model_output.a_pred, model_input.sel_actions)
        losses['no_aux_mse'] = torch.nn.MSELoss()(model_output.a_pred_no_aux, model_input.sel_actions)

        # compute total loss
        losses.total_loss = torch.stack(list(losses.values())).sum()
        return losses

    def _log_outputs(self, model_output, inputs, losses, step, phase):
        loss_reduction = losses['mse'] - losses['no_aux_mse']
        self._logger.log_scalar(loss_reduction, 'loss_reduction_negative_is_better', step)

        video_freq = 200

        inputs.current_imgs = select_indices(inputs.images.squeeze(), self.sel_times)
        inputs.current_imgs = select_indices(inputs.images.squeeze(), self.sel_times)
        if step % video_freq == 0:
            self._logger.log_kbest_videos(model_output, inputs, losses, step, phase)


class SPTModelLearnPairs(SPTModel):
    """Semi parametric transfer model"""
    def __init__(self, overrideparams, logger=None):
        super().__init__(overrideparams, logger)
        self._hp = self._default_hparams()
        self._override_defaults(overrideparams)  # override defaults with config file
        self.embeddign_dataset = None

    def _default_hparams(self):
        default_dict = AttrDict(
        )
        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def set_embedding_dataset(self, dataset):
        self.embeddign_dataset = dataset

    def build_network(self):
        super().build_network()
        self.single_state_embedding_pred = BaseProcessingNet(self._hp.state_dim, 128,
                                                  self._hp.action_dim, num_layers=3)

    def encode_sel_timestep(self, action_pred_input):
        return self.single_state_embedding_pred(action_pred_input)

    # def get_nn(self, inputs, enc_selected_t):
    #     enc_data = self.embeddign_dataset.embeddings
    #     delta_hat = np.mean(enc_selected_t[None]*enc_data, axis=1)
    #
    #     best_ind = np.argsort(delta_hat)[:self._hp.n_best]
    #     inputs.filenames[]

    # def predict_actions(self, action_pred_input, best_match_embed):




class SPTModelTest(SPTModel):
    def __init__(self, overridparams, logger=None):
        super().__init__(overridparams, logger)
        self._restore_params()

    def forward(self, inputs):
        self.T = inputs.best_matches_actions.shape[1]
        self.bsize = 1
        return self.forward_pass(inputs)
