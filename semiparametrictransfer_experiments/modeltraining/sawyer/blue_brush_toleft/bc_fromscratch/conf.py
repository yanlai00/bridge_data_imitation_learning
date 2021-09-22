import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset

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
            name='blue_brush_toleft',
            color_augmentation=0.3,
            random_crop=[48, 64],
            image_size_beforecrop=[56, 72],
            data_dir=os.environ['DATA'] + '/sawyerdata/annies_data/blue_brush_toleft/clone'
        ),
        # validation datasets:
        val0=AttrDict(
            dataclass=FixLenVideoDataset,
            dataconf=AttrDict(
                name='blue_brush_toleft_cam1',
                sel_camera=1,
                random_crop=[48, 64],
                image_size_beforecrop=[56, 72],
                data_dir=os.environ['DATA'] + '/sawyerdata/annies_data/blue_brush_toleft/clone'
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
