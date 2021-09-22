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
}

agent = {
    'type': BenchmarkAgent,
    'env': (AutograspEnv, env_params),
    'T': 30,
    'image_height': 56,
    'image_width': 72,
}

policy = {
    'type': GCBCPolicyImages,
    'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/sawyer/pickup_drill/bc_fromscratch/noclutter/weights/weights_ep382.pth',
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/sawyer/pickup_drill/bridge_targetfinetune/noclutbrid/finetuning/weights/weights_ep286.pth',
    # 'get_sub_model': 'bridge_data_params',
    # 'get_sub_model': 'single_task_params',
    'model_override_params': {
        'data_conf': {
            'data_dir': os.environ['DATA'] + '/sawyerdata/robonet_style_data/pickup_drill_wood_background',
            'random_crop':[48, 64],
            'image_size_beforecrop':[56, 72],
        },
        'sel_camera': 1,
        'predict_future_actions': False,
        'strict_loading': False
    }
}

config = {
    'current_dir' : current_dir,
    'start_index':0,
    'end_index': 100,
    'agent': agent,
    'policy': policy,
    'save_data': False,
}
