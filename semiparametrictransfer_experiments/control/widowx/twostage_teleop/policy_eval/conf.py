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


env_params = {
    'robot_name':'widowx250s',
    'robot_type':'widowx250s',
    'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw')],
    'gripper_attached': 'custom',
    'camera_server': True,
    'move_duration': 0.3,
}

agent = {
    'type': GeneralAgent,
    'env': (WidowX250sEnv, env_params),
    'T': 40,
    'image_height': 56,  # beceause of center crop
    'image_width': 72,
    # 'make_final_gif': False,
    'video_format': 'mp4',
    'recreate_env': (False, 1),
}

policy = {
    'type': GCBCPolicyImages,
    # 'restore_path': '/mount/harddrive/experiments/brc/2020-11-23/spt_experiments/modeltraining/widowx/real/put_spoon_tray/bc_fromscratch/bc_s0/weights/weights_itr180694.pth',
    'restore_path': '/mount/harddrive/experiments/spt_experiments/modeltraining/widowx/real/put_spoon_tray/bc_fromscratch/weights/weights_itr186961.pth',
    # 'confirm_first_image': True,
    'model_override_params': {
        'data_conf': {
            'data_dir' : os.environ['DATA'] + '/spt_trainingdata/control/widowx/2stage_teleop/stage0/clone',
            'random_crop':[48, 64],
            'image_size_beforecrop':[56, 72],
        },
    }
}

config = {
    'current_dir' : current_dir,
    'start_index':0,
    'end_index': 99,
    'agent': agent,
    'policy': policy,
    'save_data': False,
    # 'save_format': ['raw'],
}
