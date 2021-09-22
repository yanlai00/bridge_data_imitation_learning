import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
from semiparametrictransfer.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.data_sets.multi_dataset_loader import MultiDatasetLoader
from semiparametrictransfer.models.gcbc_transfer import GCBCTransfer
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
from semiparametrictransfer_experiments.modeltraining.sawyer.pickup_drill.datasetdef import pick_drill_wood_back, validation_data_conf, robonet_downsampled, pick_drill_wood_noclutter


data_config = AttrDict(
    main=AttrDict(
        dataclass=MultiDatasetLoader,
        dataconf=AttrDict(
            single_task=(
                FixLenVideoDataset,
                AttrDict(
                    # name='pick_drill_wood_cam0',
                    # **pick_drill_wood_back
                    name='pick_drill_wood_noclutter',
                    **pick_drill_wood_noclutter
                ),
            ),
            bridge_data=(
                FixLenVideoDataset,
                AttrDict(
                    name='robonet_sawyer',
                    **robonet_downsampled
                ),
            ),
        ),
        **validation_data_conf
    ),
    finetuning=AttrDict(
        dataclass=FixLenVideoDataset,
        dataconf=AttrDict(
            # name='pick_drill_wood_cam0',
            # **pick_drill_wood_back
            name='pick_drill_wood_noclutter',
            **pick_drill_wood_noclutter
        ),
        **validation_data_conf
    )
)

configuration = AttrDict(
    model=GCBCTransfer,
    finetuning_model=GCBCImages,
    batch_size=8,
    max_iterations=150000,
    finetuning_max_iterations=150000
    # max_iterations=15,
    # finetuning_max_iterations=15
)

model_config = AttrDict(
    main=AttrDict(
        datasource_class_mult=1,
        shared_params=AttrDict(
            action_dim=4,
            state_dim=4,
            resnet='resnet34',
            encoder_embedding_size=128,
            encoder_spatial_softmax=False,
        ),
        single_task_params=AttrDict(
            model_loss_mult=0.1
        ),
        bridge_data_params=AttrDict(
            goal_cond=True,
            sample_camera=True,
            num_domains=5
        ),
    ),
    finetuning=AttrDict(
        action_dim=4,
        state_dim=4,
        resnet='resnet34',
        encoder_embedding_size=128,
        encoder_spatial_softmax=False,
    )
)
