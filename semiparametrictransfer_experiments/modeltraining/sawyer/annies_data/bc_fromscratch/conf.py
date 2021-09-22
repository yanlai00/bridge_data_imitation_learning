import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset

# datadir = os.environ['DATA'] + '/spt_trainingdata' + '/realworld/boxpushing'       # 'directory containing data.' ,
data_dir = os.environ['DATA'] + '/spt_trainingdata' + '/realworld/can_pushing_line'       # 'directory containing data.' ,

configuration = AttrDict(
    model=GCBCImages,
    finetuning_model=GCBCImages,
    batch_size=8,
    max_iterations=200000,
)

data_config = AttrDict(
    main=AttrDict(
        dataclass=FixLenVideoDataset,
        dataconf=AttrDict(
            name='annies_data',
            T=15,
            color_augmentation=0.3,
            data_dir=os.environ['DATA'] + '/sawyerdata/annies_data/kinesthetic_demos'
        ),
        # validation datasets:
        val0=AttrDict(
            dataclass=FixLenVideoDataset,
            dataconf=AttrDict(
                name='annies_data_additionaldemos_cam0',
                # sel_camera=0,
                T=15,
                data_dir=os.environ['DATA'] + '/sawyerdata/annies_data/blue_brush_toleft/clone'
            )
        ),
        val1=AttrDict(
            dataclass=FixLenVideoDataset,
            dataconf=AttrDict(
                name='annies_data_additionaldemos_cam1',
                sel_camera=1,
                T=15,
                data_dir=os.environ['DATA'] + '/sawyerdata/annies_data/kinesthetic_demos'
            )
        )
    )
)

model_config = AttrDict(
    main=AttrDict(
        action_dim=4,
        state_dim=4,
        goal_state_delta_t=None,
        goal_cond=True
    )
)
