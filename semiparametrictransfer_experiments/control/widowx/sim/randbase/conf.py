import os.path
import experiments.control.widowx.sim.conf as base
from visual_mpc.envs.pybullet_envs.rand_robotbase_env import Widow250ContainerRandBase

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

env_params = base.env_params
env_params.pop('base_locations_file', None)
# env_params['base_locations_file'] = 'base_locations_50bases.pkl'
env_params['gui'] = False

agent = base.agent
agent['env'] = (Widow250ContainerRandBase, env_params)
policy = base.policy

config = base.config
config.update({
    'current_dir': current_dir,
    'end_index': 5000,
})
