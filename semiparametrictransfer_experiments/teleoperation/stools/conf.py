from semiparametrictransfer.utils.general_utils import AttrDict
import os
from visual_mpc.envs.robot_envs.util.multicam_server_rospkg.src.topic_utils import IMTopic
config = AttrDict(
    save_dir='/mount/harddrive/surgical_tools/pickuptools',
    randomize_initpos='line',
    T=10000,
    save_mp4=True,
    step_duration=0.1,
    enable_rotation='6dof',
    action_scaling=0.5,
    topics=[IMTopic('/cam0/image_raw')],
    custom_griper_controller=True
)
