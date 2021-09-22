from semiparametrictransfer.utils.general_utils import AttrDict
import os

config = AttrDict(
    save_dir=os.environ['DATA'] + '/spt_trainingdata/realworld/2state_looping/put_spoon_intray',
    T=500,
    save_mp4=True,
    custom_griper_controller=True,
    enable_rotation='6dof'
)
