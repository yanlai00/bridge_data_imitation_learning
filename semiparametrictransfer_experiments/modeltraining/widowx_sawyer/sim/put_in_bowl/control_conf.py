import os.path
from visual_mpc.envs.pybullet_envs.container_env import Widow250Container
from visual_mpc.envs.pybullet_envs.container_env import SawyerContainer
from visual_mpc.agent.benchmarking_agent import BenchmarkAgentPyBullet
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.policies.gcbc_policy import GCBCPolicyImages
import copy

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

env_params = AttrDict()
env_params['target_object_setting'] = 'gatorade'

agent = {
    'type': BenchmarkAgentPyBullet,
    'T': 40,
    'image_height': 56,  # beceause of random crops
    'image_width': 72,
    'recreate_env': (False, 1),  # whether to generate xml, and how often
    # 'make_final_gif_freq':1,
    'make_final_gif': False,
}

policy = {
    'type': GCBCPolicyImages,
    'model_override_params': {
        'data_conf': {
            'data_dir': os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/pick_only_gatorade/500',
            'random_crop':[48, 64],
            'image_size_beforecrop':[56, 72],
        },
        'predict_future_actions': False,
    }
}

config = {
    'current_dir' : current_dir,
    'start_index': 0,
    'end_index': 99,
    # 'end_index': 1,
    'agent': agent,
    'policy': policy,
    'save_data': False,
}

control_widowx = copy.deepcopy(config)
control_sawyer = copy.deepcopy(config)

env_params = AttrDict(gui=False,
                      target_object_setting='gatorade')
control_widowx['agent']['env'] = (Widow250Container, env_params)
control_widowx['agent']['start_goal_confs'] = os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/pick_only_gatorade/startconfig/raw'

env_params = AttrDict(gui=False,
                      target_object_setting='gatorade')
control_sawyer['agent']['env'] = (SawyerContainer, env_params)
control_sawyer['agent']['start_goal_confs'] = os.environ['DATA'] + '/spt_trainingdata/control/pybullet_sawyer/pick_only_gatorade/startconfig/raw'

control_conf = AttrDict(
    pickgatorade_widowx=control_widowx,
    pickgatorade_sawyer=control_sawyer
)
