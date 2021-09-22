from semiparametrictransfer.utils.general_utils import AttrDict
import os

config = AttrDict(
    save_dir=os.environ['DATA'] + '/spt_trainingdata' + '/realworld/can_pushing_line',
    randomize_initpos='line',
)
