""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.agent.general_agent import GeneralAgent
from semiparametrictransfer.utils.general_utils import AttrDict

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from visual_mpc.envs.robot_envs.util.multicam_server_rospkg.src.topic_utils import IMTopic
from semiparametrictransfer.policies.gcbc_policy import GCBCPolicyImages
# from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.agent.general_agent import GeneralAgent
from visual_mpc.envs.robot_envs.autograsp_env import AutograspEnv
from visual_mpc.policy.teleop.teleop_policy import SpaceMousePolicy

env_params = {
    'robot_name':'vestri',
    # 'robot_type':'sawyer',
    # 'camera_topics': [IMTopic('/cam0/image_raw'),
    #                   IMTopic('/cam1/image_raw')]
    'camera_server': True,
    # 'start_at_neutral': True,
    'cleanup_rate': 1e5
}

agent = {
    'type': GeneralAgent,
    'env': (AutograspEnv, env_params),
    'T': 15,
    'recreate_env': [False, 1],
    # 'image_height': 56,
    # 'image_width': 72,
    'image_height' : 240,
    'image_width' : 320,
    'make_final_gif':False,
    'ask_traj_ok' : True,
}

policy = {
    'type': SpaceMousePolicy,
}

config = {
    'current_dir' : current_dir,
    'start_index':0,
    'end_index': 100,
    'agent': agent,
    'policy': policy,
    'save_format': ['raw'],
}
