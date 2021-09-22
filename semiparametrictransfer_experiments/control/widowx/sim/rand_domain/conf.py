import os.path
import experiments.control.widowx.sim.conf as base
from visual_mpc.envs.pybullet_envs.rand_domain_env import Widow250ContainerRandDomain

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

env_params = base.env_params

# Rand Robot Base Configs
env_params['base_locations_file'] = 'base_locations_50bases.pkl'

# Rand View Configs
env_params['object_choices'] = (['pepsi_bottle', 'beer_bottle', 'ball', 'modern_canoe', 'colunnade_top'], 2)
env_params['view_matrices_file'] = 'view_matrices_50views.pkl'

env_params['gui'] = False

agent = base.agent
agent['env'] = (Widow250ContainerRandDomain, env_params)
policy = base.policy

config = base.config
config.update({
    'current_dir': current_dir,
    'end_index': 100,
})