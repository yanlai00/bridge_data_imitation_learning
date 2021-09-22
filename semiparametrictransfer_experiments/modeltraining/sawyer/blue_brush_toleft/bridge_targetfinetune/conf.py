import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
# from semiparametrictransfer.models.gcbc_images_context import GCBCImagesContext
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
from semiparametrictransfer.data_sets.robonet_dataloader import FilteredRoboNetDataset

# datadir = os.environ['DATA'] + '/spt_trainingdata' + '/realworld/boxpushing'       # 'directory containing data.' ,
data_dir = os.environ['DATA'] + '/spt_trainingdata' + '/realworld/can_pushing_line'       # 'directory containing data.' ,

configuration = AttrDict(
    model=GCBCImages,
    finetuning_model=GCBCImages,
    batch_size=8,
    max_iterations=200000,
    finetuning_max_iterations=200000
)

data_config = AttrDict(
    main=AttrDict(
        dataclass=FilteredRoboNetDataset,
        dataconf=AttrDict(
                         name='robonet_sawyer',
                         T=15,
                         robot_list=['sawyer'],
                         train_val_split=[0.95, 0.25, 0.25],
                         # train_val_split=[0.8, 0.1, 0.1],
                         color_augmentation=0.3,
                         random_crop=True,
                         # data_dir=os.environ['DATA'] + '/misc_datasets' + '/robonet/robonet_sampler/hdf5',
                         data_dir=os.environ['DATA'] + '/misc_datasets' + '/robonet/robonet/hdf5'
        )
    ),
    finetuning=AttrDict(
        dataclass=FixLenVideoDataset,
        dataconf=AttrDict(
            name='blue_brush_toleft',
            color_augmentation=0.3,
            random_crop=[48, 64],
            image_size_beforecrop=[56, 72],
            data_dir=os.environ['DATA'] + '/sawyerdata/annies_data/blue_brush_toleft/clone'
        ),
        # validation datasets:
        val1=AttrDict(
            dataclass=FixLenVideoDataset,
            dataconf=AttrDict(
                name='blue_brush_toleft_cam1',
                color_augmentation=0.3,
                random_crop=[48, 64],
                image_size_beforecrop=[56, 72],
                sel_camera=1,
                data_dir=os.environ['DATA'] + '/sawyerdata/annies_data/blue_brush_toleft/clone'
            )
        ),
    ),
)

model_config = AttrDict(
    main=AttrDict(
        action_dim=4,
        state_dim=4,
        goal_cond=True,
        resnet='resnet34'
    ),
    finetuning=AttrDict(
        action_dim=4,
        state_dim=4,
        goal_cond=True,
        goal_state_delta_t=None,
        resnet='resnet34'
    )
)
