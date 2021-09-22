""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path

from visual_mpc.agent.general_agent import GeneralAgent
from visual_mpc.policy.random.sampler_policy import SamplerPolicy

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from semiparametrictransfer.environments.tabletop.tabletop import Tabletop
from visual_mpc.policy.cem_controllers.cem_controller_sim import CEM_Controller_Sim
from visual_mpc.policy.cem_controllers.samplers.correlated_noise import CorrelatedNoiseSampler

from visual_mpc.agent.benchmarking_agent import BenchmarkAgent

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
    'type': BenchmarkAgent,
    'env': (Tabletop, env_params),
    'T': 15,
    'recreate_env': (True, 20),  # whether to generate xml, and how often
    'make_final_gif_freq':10,
    # 'make_final_gif': False,
    'start_goal_confs': os.environ['DATA'] + '/spt_trainingdata' + '/spt_control_experiments/control/datacollection/start_goal_pairs_3obj/raw',
    'num_load_steps': 30,
}

policy = {
    'type': CEM_Controller_Sim,
    'sampler':CorrelatedNoiseSampler,
    'sampler_params': {
        'initial_std':  [0.6, 0.6, 0.3, 0.3],
    },
    # 'num_workers' :1,
    'replan_interval': 31,
    'verbose': False,
    # 'num_samples': 200,
    'iterations': 2,
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

