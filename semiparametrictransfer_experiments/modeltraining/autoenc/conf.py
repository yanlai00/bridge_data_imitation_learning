import os
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.models.spt_model import SPTModel
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.models.det_autoencoder import TrajFollowModel

configuration = AttrDict(
    model=TrajFollowModel,
    data_dir=os.environ['DATA'] + '/spt_trainingdata' + '/sim/tabletop-texture',       # 'directory containing data.' ,
    lr= 1e-3
)

data_config = AttrDict(
                sel_len=-1,
                T=31,
)

model_config = AttrDict(
    state_dim=30,
    action_dim=4,
)
