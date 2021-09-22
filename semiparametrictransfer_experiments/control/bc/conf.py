from semiparametrictransfer_experiments.control.base_configs import table_push_base
import os

from semiparametrictransfer.policies.bc_policy import BCPolicyStates
current_dir = os.path.dirname(os.path.realpath(__file__))

table_push_base.policy.update({
    'type': BCPolicyStates,
    'restore_path': os.environ['EXP'] + '/spt_experiments' + '/modeltraining/bc/weights/weights_ep995.pth',
    'params': {'goal_cond': False},
}
)

config = table_push_base.config
config.update({'current_dir': current_dir})

