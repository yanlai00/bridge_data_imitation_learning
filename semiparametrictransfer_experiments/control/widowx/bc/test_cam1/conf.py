from semiparametrictransfer_experiments.control.widowx.bc import conf_velocity as conf
from semiparametrictransfer_experiments.control.widowx.bc import conf
from visual_mpc.envs.robot_envs.util.multicam_server_rospkg.src.topic_utils import IMTopic
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
import os

conf.policy.update({
    'restore_path': os.environ['EXP'] + '/spt_experiments' + '/modeltraining/bc/widowx_pushing/cam1/weights/weights_ep9995.pth',
})

config = conf.config
config.update({'current_dir': current_dir})

