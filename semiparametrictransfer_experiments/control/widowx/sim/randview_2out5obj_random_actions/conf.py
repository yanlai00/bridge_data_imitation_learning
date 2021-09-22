import os.path
from visual_mpc.policy.random.sampler_policy import SamplerPolicy
import experiments.control.widowx.sim.conf as base
from visual_mpc.envs.pybullet_envs.random_view_env import Widow250ContainerRandView

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

env_params = base.env_params
env_params['object_choices'] = (['pepsi_bottle', 'beer_bottle', 'ball', 'modern_canoe', 'colunnade_top'], 2)
# env_params['view_matrices_file'] = 'view_matrices_50views.pkl' ############
env_params['gui'] = False

agent = base.agent
agent['env'] = (Widow250ContainerRandView, env_params)
agent.pop('rejection_sample')

policy = {
    'type': SamplerPolicy,
    'sampler_params': {
        'initial_std':  [0.06, 0.06, 0.03, 0.0, 0.0, 0.0, 1],
    }
}


config = base.config
config.update({
    'current_dir': current_dir,
    'end_index': 5000,
    'policy': policy
})
