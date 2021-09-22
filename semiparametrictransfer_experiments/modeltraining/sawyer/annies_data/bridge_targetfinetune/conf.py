import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.models.gcbc import GCBCModel
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.multi_dataset_loader import MultiDatasetLoader
from semiparametrictransfer.models.gcbc_transfer import GCBCTransfer
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
from semiparametrictransfer.data_sets.robonet_dataloader import FilteredRoboNetDataset

# datadir = os.environ['DATA'] + '/spt_trainingdata' + '/realworld/boxpushing'       # 'directory containing data.' ,
data_dir = os.environ['DATA'] + '/spt_trainingdata' + '/realworld/can_pushing_line'       # 'directory containing data.' ,

configuration = AttrDict(
    model=GCBCImages,
    finetuning_model=GCBCImages,
    batch_size=8,
    max_iterations=300000,
    finetuning_max_iterations=200000
)

data_config = AttrDict(
    main=AttrDict(
        dataclass=FilteredRoboNetDataset,
        dataconf=AttrDict(
                         name='robonet_sawyer',
                         T=15,
                         robot_list=['sawyer'],
                         train_val_split=[0.9, 0.05, 0.05],
                         color_augmentation=0.3,
                         random_crop=True,
                         # data_dir=os.environ['DATA'] + '/misc_datasets' + '/robonet/robonet_sampler/hdf5',
                         data_dir=os.environ['DATA'] + '/misc_datasets' + '/robonet/robonet/hdf5'
        )
    ),
    finetuning=AttrDict(
        dataclass=FixLenVideoDataset,
        dataconf=AttrDict(
                     name='annies_data',
                     T=15,
                     color_augmentation=0.3,
                     data_dir= os.environ['DATA'] + '/sawyerdata/annies_data/kinesthetic_demos'
        ),
        # validation datasets:
        val1=AttrDict(
            dataclass=FixLenVideoDataset,
            dataconf=AttrDict(
                name='annies_data_additionaldemos_cam0',
                sel_camera=0,
                T=15,
                data_dir=os.environ['DATA'] + '/sawyerdata/annies_data/anniesdata_additional_demos'
            )
        ),
        val2=AttrDict(
            dataclass=FixLenVideoDataset,
            dataconf=AttrDict(
                name='annies_data_additionaldemos_cam1',
                sel_camera=1,
                T=15,
                data_dir=os.environ['DATA'] + '/sawyerdata/annies_data/anniesdata_additional_demos'
            )
        )
    ),
)

model_config = AttrDict(
    main=AttrDict(
        action_dim=4,
        state_dim=4,
        goal_cond=True,
        # resnet='resnet34'
    ),
    finetuning=AttrDict(
        action_dim=4,
        state_dim=4,
        goal_cond=True,
        goal_state_delta_t=None,
        # resnet='resnet34'
    )
)
