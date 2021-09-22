from semiparametrictransfer.utils.general_utils import AttrDict
import torch
from semiparametrictransfer.models.gcbc_images import GCBCImages
from torch.distributions.categorical import Categorical
from semiparametrictransfer.utils.general_utils import select_indices
import torch.nn as nn


def get_one_hot_sequence(nb_digits, active_dim):
    """
    param: active_dim: B, T tensor with indices that need to be 1
    """
    active_dim = active_dim.type(torch.LongTensor)
    T = active_dim.shape[1]
    batch_size = active_dim.shape[0]
    y_onehot = torch.FloatTensor(batch_size, T, nb_digits)
    y_onehot.zero_()
    y_onehot.scatter_(2, active_dim[:, :, None], 1)
    return y_onehot


class GCBCImagesDiscrete(GCBCImages):
    """Semi parametric transfer model"""

    def __init__(self, overrideparams, logger=None):
        super().__init__(overrideparams, logger)
        self._hp = self._default_hparams()
        self._override_defaults(overrideparams)  # override defaults with config file

    def discretize_actions(self, actions, pivots):
        self.num_disc = pivots.shape[1] + 1
        self.n_xy = self.num_disc**2
        self.n_z = self.num_disc
        self.n_theta = self.num_disc

        B = actions.shape[0]
        input_adim = actions.shape[2]
        T = actions.shape[1]

        assert input_adim == 4, "only supports [x,y,z,theta] action space for now!"
        assert len(pivots) == input_adim, "bad discretization pivots array!"
        binned_actions = []
        for a in range(input_adim):
            binned_action = torch.zeros((B, T), dtype=torch.int32)

            for p in range(len(pivots[a])):
                pivot = pivots[a][p]
                binned_action[actions[:, :, a] > pivot] = p

            binned_actions.append(binned_action)

        self.xy_act = binned_actions[0] + self.num_disc * binned_actions[1]
        self.z_act, self.theta_act = binned_actions[2], binned_actions[3]

        xy_one_hot = get_one_hot_sequence(self.n_xy, self.xy_act)
        z_one_hot = get_one_hot_sequence(self.n_z, self.z_act)
        theta_one_hot = get_one_hot_sequence(self.n_theta, self.theta_act)
        return torch.cat([xy_one_hot, z_one_hot, theta_one_hot], dim=2)

    def sample_tsteps(self, images, actions, pad_mask, sample_goal):


    def sample_continuous_actions(self, disc_actions, means):
        xy_logits = disc_actions[:self.n_xy]
        z_logits = disc_actions[self.n_xy:self.n_xy + self.n_z]
        theta_logits = disc_actions[self.n_xy + self.n_z:]

        xy_indices = self.sample_categorical(xy_logits)
        x_index, y_index = xy_indices % self.num_disc,  xy_indices//self.num_disc
        z_index = self.sample_categorical(z_logits)
        theta_index = self.sample_categorical(theta_logits)

        action = torch.cat([
            select_indices(means[0], x_index),
            select_indices(means[1], y_index),
            select_indices(means[2], z_index),
            select_indices(means[3], theta_index),
        ])
        return action

    def sample_categorical(self, logits):
        indices = []
        for b in range(self._hp.batch_size):
            m = Categorical(logits=logits[b])
            indices.append(m.sample())
        return torch.cat(indices, dim=0)

    def loss(self, model_input, model_output, compute_total_loss=True):
        a_pred = model_output.normed_pred_actions
        losses = AttrDict()

        losses.xy_classification_loss = [nn.CrossEntropyLoss()(a_pred, self.xy_act),
                                      self._hp.robot_class_mult]


        # compute total loss
        if compute_total_loss:
            losses.total_loss = torch.stack([l[0] * l[1] for l in losses.values()]).sum()
        return losses








