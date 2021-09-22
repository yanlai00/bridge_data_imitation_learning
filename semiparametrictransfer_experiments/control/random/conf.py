from semiparametrictransfer_experiments.control.base_configs import table_push_base
from visual_mpc.policy.random.sampler_policy import SamplerPolicy
import os

current_dir = os.path.dirname(os.path.realpath(__file__))

table_push_base.policy.update({
    'type': SamplerPolicy,
    'initial_std':  [0.6, 0.6, 0.3, 0.3],
}
)

config = table_push_base.config
config.update({'current_dir': current_dir})

