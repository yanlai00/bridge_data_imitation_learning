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
    'T': 30,
    # 'image_height': 48,
    # 'image_width': 64,
    'load_goal_image': ['/home/febert/Documents/trainingdata/spt_trainingdata/realworld/sawyer/anniesdata_additional_demos/_2020-10-05_11-23-10/raw/traj_group0/traj0/', 14]   # blue shovel
    # 'load_goal_image': ['/home/febert/Documents/trainingdata/spt_trainingdata/realworld/sawyer/anniesdata_additional_demos/_2020-10-05_11-19-54/raw/traj_group0/traj0', 14]     # green broom
    # 'load_goal_image': ['/home/febert/Documents/trainingdata/spt_trainingdata/realworld/sawyer/anniesdata_additional_demos/_2020-10-05_11-17-07/raw/traj_group0/traj0', 14]     # blue wiper
    # 'load_goal_image': ['/home/febert/Documents/trainingdata/spt_trainingdata/realworld/sawyer/anniesdata_additional_demos/_2020-10-05_11-09-17/raw/traj_group0/traj1', 14]     # yellow wiper
}

policy = {
    'type': GCBCPolicyImages,
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/sawyer/bc_fromscratch/finalstep_goal/weights/weights_ep905.pth',
    'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/sawyer/bridge_targetfinetune/finalstep_goal/finetuning/weights/weights_ep225.pth',
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/sawyer/transfer/resnet34/weights/weights_ep905.pth',
    # 'get_sub_model': 'bridge_data_params',
    'get_sub_model': 'single_task_params',
    'model_override_params': {
        'data_conf': {
            'data_dir': os.environ['DATA'] + '/sawyerdata/annies_data/kinesthetic_demos',
            'random_crop': None,
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
