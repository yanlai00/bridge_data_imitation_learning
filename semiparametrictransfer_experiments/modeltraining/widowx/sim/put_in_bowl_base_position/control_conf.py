import os.path
from visual_mpc.envs.pybullet_envs.container_env import Widow250Container
from visual_mpc.agent.benchmarking_agent import BenchmarkAgentPyBullet
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.policies.gcbc_policy import GCBCPolicyImages

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

env_params = AttrDict()
env_params['target_object_setting'] = 'gatorade'

agent = {
    'type': BenchmarkAgentPyBullet,
    'env': (Widow250Container, env_params),
    'T': 40,
    'image_height': 56,  # beceause of random crops
    'image_width': 72,
    'recreate_env': (False, 1),  # whether to generate xml, and how often
    # 'make_final_gif_freq':1,
    'make_final_gif': False,
    'start_goal_confs': os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/pick_only_gatorade/startconfig/raw'
}

policy = {
    'type': GCBCPolicyImages,
    'model_override_params': {
        'data_conf': {
            'data_dir': os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/pick_only_gatorade/5k',
            'random_crop':[48, 64],
            'image_size_beforecrop':[56, 72],
        },
        # 'predict_future_actions': False,
    }
}

config = {
    'current_dir' : current_dir,
    'start_index': 0,
    'end_index': 20,
    'agent': agent,
    'policy': policy,
    'save_data': False,
}
