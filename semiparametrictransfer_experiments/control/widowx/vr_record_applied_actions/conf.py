""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.agent.general_agent import GeneralAgent
from semiparametrictransfer.utils.general_utils import AttrDict

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from visual_mpc.envs.robot_envs.util.multicam_server_rospkg.src.topic_utils import IMTopic
from visual_mpc.envs.robot_envs.widowx250s.widowx250s_env import WidowX250sEnv
from visual_mpc.envs.robot_envs.widowx250s.widowx250s_env import VR_WidowX250S
from visual_mpc.agent.general_agent import TimedLoop
from visual_mpc.policy.vr_teleop_policy import VRTeleopPolicy

env_params = {
    'robot_name':'widowx250s',
    'robot_type':'widowx250s',
    # 'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw')],
    'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw'), IMTopic('/cam2/image_raw'), IMTopic('/cam3/image_raw'), IMTopic('/cam4/image_raw')],
    'gripper_attached': 'custom',
    'skip_move_to_neutral': True,
    'move_to_rand_start_freq': 2,
    # 'action_space':'3trans1rot'
    'move_duration': 0.1,
    'action_clipping': False
}

agent = {
    'type': TimedLoop,
    'env': (VR_WidowX250S, env_params),
    'recreate_env': (False, 1),
    'T': 100,
    'image_height': 480,
    'image_width': 640,
    'make_final_gif': False,
    'video_format': 'mp4',
}

policy = {
    'type': VRTeleopPolicy,
}

config = {
    'current_dir' : current_dir,
    'collection_metadata' : current_dir + '/collection_metadata.json',
    'start_index':0,
    'end_index': 500,
    'agent': agent,
    'policy': policy,
    # 'save_data': True,
    'save_format': ['raw'],
}
