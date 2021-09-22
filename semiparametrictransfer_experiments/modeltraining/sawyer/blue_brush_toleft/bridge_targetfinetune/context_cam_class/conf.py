import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
# from semiparametrictransfer.models.gcbc_images_context import GCBCImagesContext
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
from semiparametrictransfer.data_sets.robonet_dataloader import FilteredRoboNetDataset

from semiparametrictransfer.models.gcbc_images_context import GCBCImagesContext

configuration = AttrDict(
    model=GCBCImagesContext,
    finetuning_model=GCBCImagesContext,
    batch_size=8,
    max_iterations=150000,
    finetuning_max_iterations=150000,
)


blue_brush_to_left = AttrDict(
    color_augmentation=0.3,
    random_crop=[48, 64],
    image_size_beforecrop=[56, 72],
    data_dir=os.environ['DATA'] + '/sawyerdata/annies_data/blue_brush_toleft/clone'
)

data_config = AttrDict(
    main=AttrDict(
        dataclass=FixLenVideoDataset,
        dataconf=AttrDict(
            name='robonet_sawyer',
            color_augmentation=0.3,
            random_crop=[48, 64],
            image_size_beforecrop=[56, 72],
            data_dir=os.environ['DATA'] + '/misc_datasets/robonet/robonet/downsampled',
            sel_camera=-1
        )
    ),
    finetuning=AttrDict(
        dataclass=FixLenVideoDataset,
        dataconf=AttrDict(name='blue_brush_toleft', **blue_brush_to_left),
        # validation datasets:
        val0=AttrDict(
            dataclass=FixLenVideoDataset,
            dataconf=AttrDict(
                name='blue_brush_toleft_cam1',
                sel_camera=1,
                **blue_brush_to_left
            )
        )
    ),
)

model_config = AttrDict(
    main=AttrDict(
        action_dim=4,
        state_dim=4,
        goal_cond=True,
        resnet='resnet34',
        num_domains=5,
        domain_class_mult=1
    ),
    finetuning=AttrDict(
        action_dim=4,
        state_dim=4,
        goal_cond=True,
        goal_state_delta_t=None,
        resnet='resnet34',
    )
)
