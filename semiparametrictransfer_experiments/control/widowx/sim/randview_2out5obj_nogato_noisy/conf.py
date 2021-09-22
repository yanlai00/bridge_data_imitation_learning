import os.path
import experiments.control.widowx.sim.conf as base
from visual_mpc.envs.pybullet_envs.random_view_env import Widow250ContainerRandView

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

env_params = base.env_params
env_params['object_choices'] = (['pepsi_bottle', 'beer_bottle', 'ball', 'modern_canoe', 'colunnade_top'], 2)
env_params['view_matrices_file'] = 'view_matrices_50views.pkl' ############
env_params['gui'] = False

agent = base.agent
agent['env'] = (Widow250ContainerRandView, env_params)
agent.pop('rejection_sample')

policy = base.policy
policy.update({
    'pick_point_noise': 0.025
})

config = base.config
config.update({
    'current_dir': current_dir,
    'end_index': 5000,
})
