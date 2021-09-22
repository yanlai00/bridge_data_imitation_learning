""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.agent.general_agent import GeneralAgent
from semiparametrictransfer.utils.general_utils import AttrDict

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from visual_mpc.envs.robot_envs.util.multicam_server_rospkg.src.topic_utils import IMTopic
from visual_mpc.envs.robot_envs.widowx250s.widowx250s_env import WidowX250sEnv, WidowX250SEnvVelocity
from visual_mpc.policy.policy import DummyPolicy
from semiparametrictransfer.policies.gcbc_policy import GCBCPolicyImages
from visual_mpc.agent.general_agent import TimedLoop

env_params = {
    'robot_name':'widowx250s',
    'robot_type':'widowx250s_velocity',
    'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw'), IMTopic('/cam2/image_raw'), IMTopic('/cam3/image_raw'), IMTopic('/cam4/image_raw')],
    'gripper_attached': 'custom',
    'num_task_stages': 1,
    'move_duration': 0.7,
    'randomize_initpos': 'restricted_space'
}

agent = {
    'type': TimedLoop,
    'env': [WidowX250SEnvVelocity, env_params],
    'recreate_env': (False, 1),
    'T': 120,
    'image_height': 480,  # beceause of center crop
    'image_width': 640,
    'make_final_gif': False,
    'video_format': 'mp4',
}

policy = {
    'type': DummyPolicy,
}

config = {
    'current_dir' : current_dir,
    'start_index':0,
    'end_index': 100,
    'agent': agent,
    'policy': policy,
    # 'save_data': True,
    'save_format': ['raw'],
}
