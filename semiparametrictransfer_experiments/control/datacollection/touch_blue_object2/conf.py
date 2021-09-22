""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path

from visual_mpc.agent.general_agent import GeneralAgent
from visual_mpc.policy.random.sampler_policy import SamplerPolicy

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from semiparametrictransfer.environments.tabletop.tabletop import Tabletop
from visual_mpc.policy.cem_controllers.cem_controller_sim import CEM_Controller_Sim
# from classifier_control.environments.sim.tabletop.tabletop import Tabletop
# from classifier_control.cem_controllers.gt_dist_controller import GroundTruthDistController

from visual_mpc.policy.cem_controllers.samplers.correlated_noise import CorrelatedNoiseSampler

from visual_mpc.agent.general_agent import GeneralAgent

env_params = {
    # resolution sufficient for 16x anti-aliasing
    'viewer_image_height': 192,
    'viewer_image_width': 256,
    'textured': True,
    # for my own code:
    'xml': "assets/sawyer_xyz/sawyer_multiobject_2cam.xml",
    'ncam': 2,
    'image_rotate': [False, True],
    'set_object_touch_goal': True
    # for Stephen's code:
    # 'ncam': 1,
}


agent = {
    'type': GeneralAgent,
    'env': (Tabletop, env_params),
    'T': 15,
    'recreate_env': (True, 20),  # whether to generate xml, and how often
    # 'make_final_gif_freq':10,
    # 'make_final_gif': False,
}

policy = {
    # 'type': GroundTruthDistController,
    'type': CEM_Controller_Sim,
    'sampler':CorrelatedNoiseSampler,
    'sampler_params': {
        'initial_std':  [0.6, 0.6, 0.3, 0.3],
    },
    # 'num_workers' :1, ##########
    'replan_interval': 31,
    # 'verbose': False,
    # 'num_samples': 200, ############
    'iterations': 2,
    ##### for Stephen's code:
    # 'use_gt_model': True,
    # 'touch_object_cost':True
}

config = {
    'traj_per_file':1,  #28,
    'current_dir' : current_dir,
    'start_index':0,
    'end_index': 1000,
    'agent': agent,
    'policy': policy,
    # 'save_data': False,
    # 'save_format': ['hdf5', 'raw'],
}

