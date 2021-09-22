import os
from semiparametrictransfer.models.gcbc_images import GCBCImages
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.models.gcbc import GCBCModel
current_dir = os.path.dirname(os.path.realpath(__file__))

# datadir = os.environ['DATA'] + '/spt_trainingdata' + '/realworld/boxpushing'       # 'directory containing data.' ,
data_dir = os.environ['DATA'] + '/spt_trainingdata' + '/realworld/can_pushing_line'       # 'directory containing data.' ,

configuration = AttrDict(
    model=GCBCImages,
    batch_size=8,
    val_every_n=20,
    num_epochs=10000
)


data_config = AttrDict(
    T=30,
    color_augmentation=0.3,
    random_crop=[48, 64],
    image_size_beforecrop=[56, 72],
    data_dir=data_dir,
    camera='all',
)

model_config = AttrDict(
    action_dim=4,
    state_dim=4,
    # freeze_resnet=False,
    # resnet=None,
    # freeze_encoder=True
    # sel_camera=1,
    finetuning_override=AttrDict(
        sel_camera=1,
    )
)
