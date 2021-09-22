import os.path
import experiments.control.widowx.sim.conf as base
from visual_mpc.envs.pybullet_envs.random_view_env import Widow250ContainerRandView

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

env_params = base.env_params
env_params['target_object_setting'] = 'gatorade'
env_params['gui'] = False

agent = base.agent
agent['env'] = (Widow250ContainerRandView, env_params)

policy = base.policy

config = base.config
config.update({
    'current_dir': current_dir,
    # 'end_index': 500,
    'end_index': 5000,
})
