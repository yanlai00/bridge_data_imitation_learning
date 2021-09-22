from semiparametrictransfer.utils.general_utils import AttrDict
import os

config = AttrDict(
    save_dir=os.environ['DATA'] + '/spt_trainingdata' + '/realworld/robonetv2',
    T=180,
    save_mp4=True,
    enable_rotation='6dof'
)
