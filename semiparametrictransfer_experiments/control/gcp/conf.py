from semiparametrictransfer_experiments.control.base_configs import table_push_base
import os

from semiparametrictransfer.policies.bc_policy import BCPolicyFollowTraj
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.policies.gcp_policy import GCPPolicyStates

gcbc_params = {
    'restore_path' :os.environ['EXP'] + '/spt_experiments' + '/modeltraining/gcbc/invm3/weights/weights_ep995.pth'
}

gcp_params = {
    'restore_path' :os.environ['EXP'] + '/spt_experiments' + '/modeltraining/gcp/normalized/weights/weights_ep995.pth'
}

table_push_base.policy.update({
    'type': GCPPolicyStates,
    'gcbc_params': gcbc_params,
    'gcp_params': gcp_params,
})


config = table_push_base.config
config.update({'current_dir': current_dir})

