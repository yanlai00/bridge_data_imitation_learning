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
    'robot_type':'widowx250s',
    'camera_topics': [IMTopic('/cam0/image_raw')],
    'gripper_attached': 'custom',
    # 'camera_server': True,
    # 'move_duration': 0.3, # by default!
    'move_duration': 0.1,
    'action_clipping': False
}

agent = {
    'type': TimedLoop,
    'env': (WidowX250sEnv, env_params),
    'T': 90,
    'image_height': 112,  # beceause of center crop
    'image_width': 144,
    # 'make_final_gif': False,
    'video_format': 'mp4',
    'recreate_env': (False, 1),
}

policy = {
    'type': GCBCPolicyImages,
    # 'restore_path': '/mount/harddrive/experiments/brc/2021-01-20/spt_experiments/modeltraining/widowx/real/viewpoint_invariance/bc_fromscratch/bc/weights/weights_itr128029.pth',
    'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/viewpoint_invariance_pickup/bc_fromscratch/sep_tsteps/weights/weights_itr196352.pth',
    'confirm_first_image': True,
    'model_override_params': {
        'data_conf': {
            # 'data_dir' : os.environ['DATA'] + '/spt_trainingdata/control/widowx/2stage_teleop/robonet_lowres/stage0/clone/hdf5',
            # 'data_dir' : os.environ['DATA'] + '/spt_trainingdata/control/widowx/vr_control/bww_grasp_pen/hdf5',
            'data_dir' : os.environ['DATA'] + '/spt_trainingdata/control/widowx/vr_record_applied_actions/bww_grasp_pen/hdf5',
            'random_crop':[96, 128],
            'image_size_beforecrop':[112, 144]
        },
        'img_sz': [96, 128],
        'sel_camera': 0
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