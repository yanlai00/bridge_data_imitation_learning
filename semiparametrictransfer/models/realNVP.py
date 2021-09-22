import torch
import torch.nn as nn
from torch import distributions
import numpy as np
from semiparametrictransfer.utils.general_utils import npy2trch

class ObservationConditionedRealNVP(nn.Module):
    """
    Adapted from: https://github.com/papercup-open-source/tutorials/blob/master/intro_nf/nf_tutorial_torch.ipynb
    Changes:
        Works with N-D data (instead of just 2D)
        Coupling layers are conditioned on the current observation
    """
    def __init__(
        self,
        flips,
        action_dim,
        obs_processor,
        ignore_observation=False,
        use_atanh_preprocessing=False,
    ):
        super().__init__()

        # for the flipping part of the code to work correctly
        assert action_dim % 2 == 0

        self.action_dim = action_dim
        self.flips = flips
        self.obs_processor = obs_processor
        self.num_obs_features = obs_processor.output_size
        self.ignore_observation = ignore_observation
        self.use_atanh_preprocessing = use_atanh_preprocessing

        self.prior = distributions.MultivariateNormal(torch.zeros(action_dim),
                                                      torch.eye(action_dim))
        self.shift_log_scale_fns = nn.ModuleList()

        if self.ignore_observation:
            shift_log_scale_input_size = action_dim//2
            self.obs_processor = None
        else:
            shift_log_scale_input_size = action_dim//2 + self.num_obs_features
        for _ in flips:
            shift_log_scale_fn = nn.Sequential(
                nn.Linear(shift_log_scale_input_size, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
            )
            self.shift_log_scale_fns.append(shift_log_scale_fn)

    def get_action(self, obs_np, x=None):
        if x is None:
            x = npy2trch(np.random.normal(size=(1, self.action_dim)))
        else:
            x = npy2trch(np.expand_dims(x, axis=0))

        if isinstance(obs_np, dict):
            obs = {}
            for k in obs_np.keys():
                obs[k] = npy2trch(np.expand_dims(obs_np[k], axis=0))
        else:
            obs = npy2trch(np.expand_dims(obs_np, axis=0))

        for i, _ in enumerate(self.flips):
            x = self.bijector_forward(x, obs, flip_idx=i)
        x = x.squeeze()

        if self.use_atanh_preprocessing:
            action = torch.tanh(x)
        else:
            action = x

        return ptu.get_numpy(action), {}

    def bijector_forward(self, x, observation, flip_idx):
        """
        :param x: noise, of shape [batch_size, output_size]
        :param observation:
        :param flip_idx: whether to apply the flip or not
        """

        flip = self.flips[flip_idx]
        d = x.shape[-1] // 2
        x1, x2 = x[:, :d], x[:, d:]
        if flip:
            x2, x1 = x1, x2

        if not self.ignore_observation:
            observation_features = self.obs_processor(observation)
            net_out = self.shift_log_scale_fns[flip_idx](
                torch.cat((observation_features, x1), 1)
            )
            self.observation_features = observation_features
        else:
            net_out = self.shift_log_scale_fns[flip_idx](x1)
        shift = net_out[:, :self.action_dim // 2]
        log_scale = torch.tanh(net_out[:, self.action_dim // 2:])
        scale = torch.exp(log_scale)
        if torch.isnan(scale).any():
            raise RuntimeError('Scale factor has NaN entries in forward flow')
        y2 = x2 * scale + shift
        if flip:
            x1, y2 = y2, x1
        y = torch.cat([x1, y2], -1)
        return y

    def bijector_inverse_forward(self, y, obs, flip_idx):
        flip = self.flips[flip_idx]
        d = y.shape[-1] // 2
        y1, y2 = y[:, :d], y[:, d:]
        if flip:
            y1, y2 = y2, y1

        if not self.ignore_observation:
            obs_feats = self.obs_processor(obs)
            net_out = self.shift_log_scale_fns[flip_idx](
                torch.cat((obs_feats, y1), 1))
        else:
            net_out = self.shift_log_scale_fns[flip_idx](y1)

        shift = net_out[:, :self.action_dim // 2]
        log_scale = torch.tanh(net_out[:, self.action_dim // 2:])
        scale = torch.exp(-log_scale)
        if torch.isnan(scale).any():
            raise RuntimeError('Scale factor has NaN entries in reverse flow')
        x2 = (y2 - shift) * scale
        if flip:
            y1, x2 = x2, y1
        x = torch.cat([y1, x2], -1)
        # Summing instead of multiplying since working in log space
        # TODO(avi) check correctness
        return x, torch.sum(log_scale, 1)

    def log_prob_chain(self, y, obs):
        # Run y through all the necessary inverses, keeping track
        # of the logscale along the way, allowing us to compute the loss.
        def atanh(x):
            return 0.5 * torch.log((1 + x) / (1 - x))

        if self.use_atanh_preprocessing:
            temp = atanh(y)
        else:
            temp = y

        logscales = y.data.new(y.shape[0]).zero_()
        for i, _ in enumerate(self.flips):
            temp, logscale = self.bijector_inverse_forward(
                temp,
                obs,
                flip_idx=len(self.flips) - 1 - i,
            )
            # One logscale per element in a batch per layer of flow.
            logscales += logscale.squeeze(-1)
        return self.base_log_prob_fn(temp) - logscales

    @staticmethod
    def base_log_prob_fn(x):
        return torch.sum(- (x ** 2) / 2 - np.log(np.sqrt(2 * np.pi)), -1)
