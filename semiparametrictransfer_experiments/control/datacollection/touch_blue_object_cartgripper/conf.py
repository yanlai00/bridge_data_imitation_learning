""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path

from visual_mpc.agent.general_agent import GeneralAgent

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from visual_mpc.policy.random.sampler_policy import SamplerPolicy
# from classifier_control.cem_controllers.gt_dist_controller import GroundTruthDistController
from visual_mpc.policy.cem_controllers.samplers.correlated_noise import CorrelatedNoiseSampler

from  visual_mpc.envs.mujoco_env.cartgripper_env.cartgripper_xyz_grasp import CartgripperXY3BlockEnv

from visual_mpc.agent.general_agent import GeneralAgent

env_params = {
    'ncam': 2,
    'clean_xml': False,
    'num_objects': 3,
    'arm_start_lifted': False
}


agent = {
    'type': GeneralAgent,
    'env': (CartgripperXY3BlockEnv, env_params),
    'T': 15,
    'recreate_env': (True, 20),  # whether to generate xml, and how often
    # 'make_final_gif_freq':10,
    # 'make_final_gif': False,
}

policy = {
    'type': SamplerPolicy,
    'sampler_params': {
        'initial_std':  [0.06, 0.06, 0.0, 0.1],
        # 'initial_std': [0.06, 0.06],
    }
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

