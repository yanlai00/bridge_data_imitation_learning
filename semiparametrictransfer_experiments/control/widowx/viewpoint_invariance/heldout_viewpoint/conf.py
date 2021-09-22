""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
# from visual_mpc.agent.general_agent import GeneralAgent
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
    # 'move_duration': 0.3, # by default!
    'randomize_initpos': 'restricted_space'
}

agent = {
    'type': BenchmarkAgent,
    'env': (WidowX250sEnv, env_params),
    'T': 40,
    'image_height': 112,  # beceause of center crop
    'image_width': 144,
    # 'make_final_gif': False,
    'video_format': 'mp4',
    'recreate_env': (False, 1),
    # 'load_goal_image': '/mount/harddrive/trainingdata/spt_trainingdata/control/widowx/put_knive_in_tray_5cam/2021-01-25_19-31-27/raw/traj_group0/traj0/images2/im_15.png',
    'load_goal_image': '/mount/harddrive/trainingdata/spt_trainingdata/control/widowx/put_fork_in_tray_5cam/2021-01-25_15-16-06/raw/traj_group0/traj4/images2/im_23.png',
}

policy = {
    'type': GCBCPolicyImages,
    # 'restore_path': '/mount/harddrive/experiments/brc/2021-01-20/spt_experiments/modeltraining/widowx/real/viewpoint_invariance/transfer/camera_class/1tr_cl0.1/weights/weights_itr149160.pth',
    'restore_path': '/mount/harddrive/experiments/spt_experiments/modeltraining/widowx/real/viewpoint_invariance/transfer/camera_class/200ex/weights_itr106898.pth',
    'confirm_first_image': True,
    'get_sub_model': 'bridge_data_params',
    'model_override_params': {
        'data_conf': {
            'data_dir' : os.environ['DATA'] + '/spt_trainingdata/control/widowx/2stage_teleop/robonet_lowres/stage0/clone/hdf5',
            'random_crop':[96, 128],
            'image_size_beforecrop':[112, 144]
        },
        'img_sz': [96, 128],
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