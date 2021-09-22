import os
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.models.gcbc import GCBCModel
current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = AttrDict(
    model=GCBCModel,
    data_dir=os.environ['DATA'] + '/spt_trainingdata' + '/sim/tabletop-texture',       # 'directory containing data.' ,
    # batch_size=32,
)


data_config = AttrDict(
                sel_len=-1,
                T=31,
)

model_config = AttrDict(
    state_dim=30,
    action_dim=4,
    goal_state_delta_t=3
)
