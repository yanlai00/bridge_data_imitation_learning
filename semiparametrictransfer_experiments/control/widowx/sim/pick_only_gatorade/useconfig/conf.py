import os.path
import experiments.control.widowx.sim.conf as base

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))
from visual_mpc.agent.benchmarking_agent import BenchmarkAgentPyBullet

env_params = base.env_params
env_params['target_object_setting'] = 'gatorade'

agent = base.agent
agent.update({
    'type': BenchmarkAgentPyBullet,
    'start_goal_confs': '/mount/harddrive/trainingdata/spt_trainingdata/control/widowx/sim/pick_only_gatorade/startconfig/raw'
})
policy = base.policy


config = base.config
config.update({
    'current_dir': current_dir,
    'end_index': 100,
    'save_format': ['raw'],
})
