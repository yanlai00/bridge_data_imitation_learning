import os.path
import experiments.control.pybullet_sawyer.rand2out6.conf as base
from semiparametrictransfer.utils.general_utils import AttrDict
from visual_mpc.envs.pybullet_envs.container_env import SawyerContainer

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

env_params = AttrDict(
    target_object_setting='gatorade',
    gui=False
)

agent = base.agent
agent['env'] = (SawyerContainer, env_params)

policy = base.policy

config = base.config
config.update({
    'current_dir': current_dir,
    'end_index': 100,
    'save_format': ['raw'],
})
