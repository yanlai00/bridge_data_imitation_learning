from semiparametrictransfer_experiments.control.base_configs import table_push_base
import os

from semiparametrictransfer.policies.bc_policy import BCPolicyFollowTraj
current_dir = os.path.dirname(os.path.realpath(__file__))

table_push_base.policy.update({
    'type': BCPolicyFollowTraj,
    'restore_path': os.environ['EXP'] + '/spt_experiments' + '/modeltraining/trajfollowmodel/weights/weights_ep845.pth',
}
)

config = table_push_base.config
config.update({'current_dir': current_dir})

