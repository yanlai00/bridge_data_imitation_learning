import os.path
from visual_mpc.agent.general_agent import GeneralAgent

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from semiparametrictransfer.environments.tabletop.tabletop import Tabletop
from visual_mpc.policy.random.sampler_policy import SamplerPolicy

env_params = {
    # resolution sufficient for 16x anti-aliasing
    'viewer_image_height': 192,
    'viewer_image_width': 256,
    'textured': True,
    'xml': "assets/sawyer_xyz/sawyer_multiobject_2cam.xml",
    'ncam': 2,
    'image_rotate': [False, True],
}


agent = {
    'type': GeneralAgent,
    'env': (Tabletop, env_params),
    'T': 30,
    'recreate_env': (True, 20),  # whether to generate xml, and how often
    # 'make_final_gif_freq':1,
    'make_final_gif':False
}


policy = {
    'type': SamplerPolicy,
    'initial_std':  [0.6, 0.6, 0.3, 0.3],
}

config = {
    'traj_per_file':1,  #28,
    'current_dir' : current_dir,
    'start_index':0,
    'end_index': 10000,
    'agent': agent,
    'policy': policy,
    # 'save_data': True,
    'save_format': ['raw', 'hdf5'],
}