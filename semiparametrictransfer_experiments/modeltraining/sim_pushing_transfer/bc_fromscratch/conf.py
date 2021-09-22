import os
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.models.gcbc_images import GCBCImages
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = AttrDict(
    model=GCBCImages,
    batch_size=8,
    max_iterations=50000
)

data_config = AttrDict(
  main=[
        FixLenVideoDataset,
                 AttrDict(
                     T=31,
                     data_dir=os.environ['DATA'] + '/spt_trainingdata' + '/spt_control_experiments/control/datacollection/gtruth_model_demo',  # 'directory containing data.' ,
                 )
         ]
)

model_config = AttrDict(
    state_dim=30,
    action_dim=4,
    goal_cond=True,
    sel_camera=1
)
