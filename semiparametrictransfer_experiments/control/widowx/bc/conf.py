""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.agent.general_agent import GeneralAgent
from semiparametrictransfer.utils.general_utils import AttrDict

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from visual_mpc.envs.robot_envs.util.multicam_server_rospkg.src.topic_utils import IMTopic
from visual_mpc.envs.robot_envs.widowx250s.widowx250s_env import WidowX250sEnv
from semiparametrictransfer.policies.gcbc_policy import GCBCPolicyImages

env_params = {
    'robot_name':'widowx250s',
    'robot_type':'widowx250s',
    'camera_topics': [IMTopic('/cam0/image_raw')],
    'gripper_attached': 'default',
    'camera_server': True,
    'randomize_initpos': 'line'
}

agent = {
    'type': GeneralAgent,
    'env': (WidowX250sEnv, env_params),
    'T': 30,
    'image_height': 56,  # beceause of center crop
    'image_width': 72,
}

policy = {
    'type': GCBCPolicyImages,
    'restore_path': os.environ['EXP'] + '/spt_experiments' + '/modeltraining/bc/widowx_pushing/can_freeze_nopretrained/weights/weights_ep9995.pth',
}

config = {
    'traj_per_file':1,  #28,
    'current_dir' : current_dir,
    'start_index':0,
    'end_index': 100,
    'agent': agent,
    'policy': policy,
    'save_data': False,
    'save_format': ['raw'],
}
