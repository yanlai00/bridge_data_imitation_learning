import os.path
from visual_mpc.agent.general_agent import GeneralAgent
from visual_mpc.agent.benchmarking_agent import BenchmarkAgentPyBullet
from semiparametrictransfer.environments.pybullet_envs.widow250 import Widow250Env
from semiparametrictransfer.environments.pybullet_envs.widow250container import Widow250Container
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.policies.scripted_policies.pickplace import PickPlace
from semiparametrictransfer.policies.gcbc_policy import GCBCPolicyImages

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

env_params = AttrDict()
env_params['target_object_setting'] = 'gatorade'

agent = {
    'type': BenchmarkAgentPyBullet,
    'env': (Widow250Container, env_params),
    'T': 50,
    'image_height': 56,  # beceause of random crops
    'image_width': 72,
    'recreate_env': (False, 1),  # whether to generate xml, and how often
    # 'make_final_gif_freq':1,
    'make_final_gif': False,
    'start_goal_confs': os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/pick_only_gatorade/startconfig/raw'
}

policy = {
    'type': GCBCPolicyImages,
    'restore_path': '/mount/harddrive/experiments/brc/2020-12-03/spt_experiments/modeltraining/widowx/sim/put_in_bowl/bc_fromscratch/10k/bc_10k/weights/weights_itr69875.pth',
    'confirm_first_image': True,
    'model_override_params': {
        'data_conf': {
            'data_dir' : os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/pick_only_gatorade/10k',
            'random_crop':[48, 64],
            'image_size_beforecrop':[56, 72],
        },
        'predict_future_actions': False,
    }
}

config = {
    'current_dir' : current_dir,
    'start_index': 0,
    'end_index': 100,
    'agent': agent,
    'policy': policy,
    'save_data': False,
}
