import os.path
from visual_mpc.agent.general_agent import GeneralAgent
from visual_mpc.envs.pybullet_envs.container_env import Widow250Container
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.policies.scripted_policies.pickplace import PickPlace

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CAMERA = AttrDict(
                    target_pos=(0.6, 0.2, -0.28),
                    distance=0.29,
                    roll=0.0,
                    pitch=-40,
                    up_axis_index=2
                )

env_params = AttrDict(
    camera_settings=[AttrDict(**DEFAULT_CAMERA, yaw=180),
                     AttrDict(**DEFAULT_CAMERA, yaw=200),
                     AttrDict(**DEFAULT_CAMERA, yaw=160)],
    # gui=False
)

agent = {
    'type': GeneralAgent,
    'env': (Widow250Container, env_params),
    'T': 30,
    'image_height': 56,  # beceause of random crops
    'image_width': 72,
    'recreate_env': (False, 1),  # whether to generate xml, and how often
    # 'make_final_gif_freq':1,
    'make_final_gif': False,
    'rejection_sample': True
}

policy = {
    'type': PickPlace
}

config = {
    'current_dir' : current_dir,
    'start_index': 0,
    'end_index': 5000,
    'agent': agent,
    'policy': policy,
    # 'save_data': False,  #true by default
    'save_format': ['hdf5'],
}
