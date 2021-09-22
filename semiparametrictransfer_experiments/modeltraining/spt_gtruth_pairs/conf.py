import os
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.models.spt_model import SPTModel
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.data_loader import GtruthPariningDataset

configuration = AttrDict(
    model=SPTModel,
    data_dir=os.environ['DATA'] + '/spt_trainingdata' + '/sim/tabletop-texture',       # 'directory containing data.' ,
    dataset_class=GtruthPariningDataset,
    # batch_size=32,
    lr=1e-3,
)

data_config = AttrDict(
                sel_len=-1,
                T=31,
                gtruth_pairings={'train':configuration.data_dir + '/gtruth_nn_train.pkl',
                                 'val':configuration.data_dir + '/gtruth_nn_val.pkl'},
                n_best=1,
                # n_best=10,
)

model_config = AttrDict(
    state_dim=30,
    action_dim=4,
    goal_cond=False,
)
