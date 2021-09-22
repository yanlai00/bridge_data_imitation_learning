""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from semiparametrictransfer.environments.tabletop.tabletop import Tabletop
from semiparametrictransfer.models.gcbc import GCBCModel
from semiparametrictransfer.policies.bc_policy import BCPolicyStates

env_params = {
    # resolution sufficient for 16x anti-aliasing
    'viewer_image_height': 192,
    'viewer_image_width': 256,
    'textured': True,
}


agent = {
    'type': BenchmarkAgent,
    'env': (Tabletop, env_params),
    'T': 30,
    'recreate_env': (True, 20),  # whether to generate xml, and how often
    # 'make_final_gif_freq':1,
    'start_goal_confs': os.environ['DATA'] + '/spt_trainingdata' + '/sim/tabletop-texture-benchmark/raw',
    'num_load_steps':30,
}


policy = {
    'type': BCPolicyStates,
    'restore_path': os.environ['EXP'] + '/spt_experiments' + '/modeltraining/gcbc/alldata/weights/weights_ep300.pth'
}

config = {
    'traj_per_file':1,  #28,
    'current_dir' : current_dir,
    'start_index':0,
    'end_index': 100,
    'agent': agent,
    'policy': policy,
    'save_data': False,
    'save_format': ['raw'],
}
