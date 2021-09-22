""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from visual_mpc.agent.general_agent import GeneralAgent
from semiparametrictransfer.utils.general_utils import AttrDict

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

import copy
from visual_mpc.envs.robot_envs.util.multicam_server_rospkg.src.topic_utils import IMTopic
from visual_mpc.envs.robot_envs.widowx250s.widowx250s_env import WidowX250sEnv
from semiparametrictransfer.policies.gcbc_policy import GCBCPolicyImages
from visual_mpc.agent.alternating_policies_agent import AlternatingPoliciesAgent


env_params = {
    'robot_name':'widowx250s',
    'robot_type':'widowx250s',
    'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw')],
    'gripper_attached': 'custom',
    'camera_server': True,
    'move_duration': 0.3,
}

agent = {
    'type': AlternatingPoliciesAgent,
    'env': (WidowX250sEnv, env_params),
    'T': 50,
    'image_height': 112,  # beceause of center crop
    'image_width': 144,
    # 'make_final_gif': False,
    'video_format': 'mp4',
    'recreate_env': (False, 1),
}

policy_base = {
    'type': GCBCPolicyImages,
    'model_override_params': {
        'data_conf': {
            'data_dir': os.environ['DATA'] + '/spt_trainingdata/control/widowx/2stage_teleop/raw/large/stage0/clone/',
            'random_crop': [96, 128],
            'image_size_beforecrop': [112, 144]
        },
        'img_sz': [96, 128]
    }
}

policy_stage0 = copy.deepcopy(policy_base)
policy_stage0['restore_path'] = '/mount/harddrive/experiments/spt_experiments/modeltraining/widowx/real/put_spoon_tray/bc_fromscratch/highres/weights/weights_itr169070.pth'
policy_stage1 = copy.deepcopy(policy_base)
policy_stage1['restore_path'] = '/mount/harddrive/experiments/spt_experiments/modeltraining/widowx/real/put_spoon_tray/bc_fromscratch/reverse/highres/moredata/weights/weights_itr105560_saved.pth'


config = {
    'current_dir' : current_dir,
    'start_index':0,
    'end_index': 99,
    'agent': agent,
    'policy': [policy_stage0, policy_stage1],
    'save_data': False,
    # 'save_format': ['raw'],
}
