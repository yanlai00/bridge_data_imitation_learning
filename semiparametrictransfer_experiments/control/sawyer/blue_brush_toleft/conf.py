""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.agent.general_agent import GeneralAgent
from semiparametrictransfer.utils.general_utils import AttrDict

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from visual_mpc.envs.robot_envs.util.multicam_server_rospkg.src.topic_utils import IMTopic
from semiparametrictransfer.policies.gcbc_policy import GCBCPolicyImages
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.envs.robot_envs.autograsp_env import AutograspEnv

env_params = {
    'robot_name':'vestri',
    # 'robot_type':'sawyer',
    # 'camera_topics': [IMTopic('/cam0/image_raw'),
    #                   IMTopic('/cam1/image_raw')]
    'camera_server': True,
    # 'start_at_neutral': True,
    'rand_drop_reset': False,
    'start_at_current_pos':True
}

agent = {
    'type': BenchmarkAgent,
    'env': (AutograspEnv, env_params),
    'T': 15,
    'image_height': 56,
    'image_width': 72,
    'load_goal_image': ['/home/febert/Documents/trainingdata/sawyerdata/annies_data/blue_brush_toleft/_2020-10-09_14-47-28/raw/traj_group0/traj1', 14]   # blue shovel
}

policy = {
    'type': GCBCPolicyImages,
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/sawyer/bc_fromscratch/weights/weights_ep19995.pth',
    'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/sawyer/bridge_targetfinetune/finetuning/weights/weights_ep9520.pth',
    # 'get_sub_model': 'bridge_data_params',
    # 'get_sub_model': 'single_task_params',
    'model_override_params': {
        'data_conf': {
            'data_dir': os.environ['DATA'] + '/sawyerdata/annies_data/blue_brush_toleft/clone',
            'random_crop':[48, 64],
            'image_size_beforecrop':[56, 72],
        },
        'sel_camera': 1
    }
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
