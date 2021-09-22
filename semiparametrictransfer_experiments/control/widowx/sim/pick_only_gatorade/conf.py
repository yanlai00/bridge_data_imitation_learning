import os.path
import experiments.control.widowx.sim.conf as base

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

env_params = base.env_params
env_params['target_object_setting'] = 'gatorade'
env_params['gui'] = False

agent = base.agent
policy = base.policy

config = base.config
config.update({
    'current_dir': current_dir,
    # 'end_index': 500,
    'end_index': 5000,
})
