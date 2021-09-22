import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
from semiparametrictransfer.data_sets.multi_dataset_loader import RandomMixingDatasetLoader

from semiparametrictransfer_experiments.modeltraining.sawyer.pickup_drill.datasetdef import pick_drill_wood_back, validation_data_conf, robonet_downsampled, pick_drill_wood_noclutter

configuration = AttrDict(
    model=GCBCImages,
    finetuning_model=GCBCImages,
    batch_size=16,
    max_iterations=150000,
    finetuning_max_iterations=150000
    # max_iterations=15,
    # finetuning_max_iterations=15
)

data_config = AttrDict(
    main=AttrDict(
        dataclass=RandomMixingDatasetLoader,
        dataconf=AttrDict(
            dataset0=[FixLenVideoDataset,
                AttrDict(
                    name='robonet_sawyer',
                        T=16,
                        color_augmentation=0.3,
                        random_crop=[48, 64],
                        image_size_beforecrop=[56, 72],
                        data_dir=os.environ['DATA'] + '/misc_datasets/robonet/robonet/downsampled',
                        sel_camera='random'
                    )
                ],
            dataset1=[FixLenVideoDataset,
                AttrDict(
                    name='pick_drill_wood_noclutter',
                    **pick_drill_wood_noclutter,
                     sel_camera=1
                    )
             ]
        )
    ),
    finetuning=AttrDict(
        dataclass=FixLenVideoDataset,
        dataconf=AttrDict(
            # name='pick_drill_wood_cam0',
            # **pick_drill_wood_back
            name='pick_drill_wood_noclutter',
            **pick_drill_wood_noclutter
        ),
        # validation datasets:
        **validation_data_conf
    ),
)

model_config = AttrDict(
    main=AttrDict(
        action_dim=4,
        state_dim=4,
        goal_cond=True,
        resnet='resnet34',
        # encoder_embedding_size=128,
        # encoder_spatial_softmax=False,
    ),
    finetuning=AttrDict(
        action_dim=4,
        state_dim=4,
        resnet='resnet34',
        # encoder_embedding_size=128,
        # encoder_spatial_softmax=False,
    )
)
