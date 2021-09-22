""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.agent.general_agent import GeneralAgent
from semiparametrictransfer.utils.general_utils import AttrDict

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from visual_mpc.envs.robot_envs.util.multicam_server_rospkg.src.topic_utils import IMTopic
from visual_mpc.envs.robot_envs.widowx250s.widowx250s_env import WidowX250sEnv, WidowX250SEnvVelocity
from semiparametrictransfer.policies.gcbc_policy import GCBCPolicyImages
from visual_mpc.agent.general_agent import TimedLoop

env_params = {
    'robot_name':'widowx250s',
    'robot_type':'widowx250s_velocity',
    'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw')],
    'gripper_attached': 'default',
    'camera_server': True,
    'randomize_initpos': 'line'
}

agent = {
    'type': TimedLoop,
    'env': (WidowX250SEnvVelocity, env_params),
    'T': 200,
    'image_height': 56,  # beceause of center crop
    'image_width': 72,
    'step_duration': 0.3
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
