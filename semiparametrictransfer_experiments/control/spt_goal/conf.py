from semiparametrictransfer_experiments.control.spt_nogoal import conf
import os

current_dir = os.path.dirname(os.path.realpath(__file__))

conf.policy.update({
    'restore_path': os.environ['EXP'] + '/spt_experiments' + '/modeltraining/spt_gtruth_pairs/pushsame/weights/weights_ep335.pth',
}
)
conf.policy.pop('params')

config = conf.config
config.update({'current_dir': current_dir})

