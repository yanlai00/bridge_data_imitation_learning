""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from semiparametrictransfer.policies.gcbc_policy import GCBCPolicyImages
from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
from widowx_envs.widowx.widowx_env import WidowXEnv
from widowx_envs.control_loops import TimedLoop

env_params = {
    # 'camera_topics': [IMTopic('/cam1/image_raw')],
    'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw'), IMTopic('/cam2/image_raw'), IMTopic('/cam3/image_raw'), IMTopic('/cam4/image_raw')],
    'gripper_attached': 'custom',
    'skip_move_to_neutral': True,
    'action_mode':'3trans',
    'fix_zangle': 0.1,
    'move_duration': 0.2,
    'override_workspace_boundaries': [[0.2, -0.04, 0.03, -1.57, 0], [0.31, 0.04, 0.1,  1.57, 0]]
}

agent = {
    'type': TimedLoop,
    'env': (WidowXEnv, env_params),
    'T': 30,
    'image_height': 56,  # beceause of center crop
    'image_width': 72,
    # 'make_final_gif': False,
    # 'video_format': 'gif',   # already by default
    'recreate_env': (False, 1),
    'ask_confirmation': False,
    'load_goal_image': [os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_blue_elephant_no_distractors/2021-04-30_15-26-34/raw/traj_group0/traj0', 17],
}

policy = {
    'type': GCBCPolicyImages,
    # 'restore_path':  os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/viewpoint_invariance_pickup_3dof/transfer_lmdbloader/shared_classifier/run2/weights/weights_itr150001.pth',
    # 'restore_path':  os.environ['EXP'] + '/brc/lmdb_real_world_05-03-21/real/viewpoint_invariance_pickup_3dof/transfer/camera_class/add_source_tobridge/weights/weights_itr150001.pth',
    'restore_path':  os.environ['EXP'] + '/brc/lmdb_real_world_05-03-21/real/viewpoint_invariance_pickup_3dof/transfer/camera_class/add_source_to_bridge/weights/weights_itr150001.pth',
    # 'get_sub_model': 'bridge_data_params',
    # 'confirm_first_image': True,
    'model_override_params': {
        'data_conf': {
            'data_dir': os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww/pick_blue_elephant_no_distractors/lmdb',
            'random_crop': [48, 64],
            'image_size_beforecrop': [56, 72]
        },
        'img_sz': [48, 64],
        'sel_camera': 4
    },
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