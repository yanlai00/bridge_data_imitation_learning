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
from visual_mpc.policy.policy import ReplayActions
from semiparametrictransfer.policies.gcbc_policy import GCBCPolicyImages
from visual_mpc.agent.general_agent import TimedLoop

# load_file = '/mount/harddrive/trainingdata/spt_trainingdata/control/widowx/vr_control/test/2021-02-26_17-24-45/raw/traj_group0/traj0'
# load_file = '/mount/harddrive/trainingdata/spt_trainingdata/control/widowx/vr_control/test6dof/2021-02-26_17-21-37/raw/traj_group0/traj0'
load_file = '/home/datacol/Documents/data/spt_trainingdata/control/widowx/vr_record_applied_actions/2021-03-06_12-39-08/raw/traj_group0/traj0'

env_params = {
    'robot_name':'widowx250s',
    'robot_type':'widowx250s',
    'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw')],
    'gripper_attached': 'custom',
    # 'camera_server': True,
    'start_state': load_file,
    # 'action_space': '3trans1rot'
}

agent = {
    'type': GeneralAgent,
    'env': (WidowX250sEnv, env_params),
    'T': 150,
    'image_height': 56,  # beceause of center crop
    'image_width': 72,
    'make_final_gif': False
}

policy = {
    'type': ReplayActions,
    'load_file': load_file
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
