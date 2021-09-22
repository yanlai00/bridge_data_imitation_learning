import os
from semiparametrictransfer_experiments.control.widowx.twostage_teleop.conf import config as base_conf
from visual_mpc.envs.robot_envs.util.multicam_server_rospkg.src.topic_utils import IMTopic
current_dir = os.path.dirname(os.path.realpath(__file__))

config = base_conf
config['agent']['env'][1] = {
    'robot_name':'widowx250s',
    'robot_type':'widowx250s_velocity',
    'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw'), IMTopic('/cam2/image_raw'), IMTopic('/cam3/image_raw'), IMTopic('/cam4/image_raw'), IMTopic('/hand/image_raw')],
    'gripper_attached': 'custom',
    'num_task_stages': 1,
    'move_duration': 0.7,
}

config['current_dir'] = current_dir

# config['agent'].pop('make_final_gif')
config['agent']['T'] = 300

