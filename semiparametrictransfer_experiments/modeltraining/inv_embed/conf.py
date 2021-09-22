from semiparametrictransfer_experiments.modeltraining.base_configs import conf
from semiparametrictransfer.utils.general_utils import AttrDict
import os
from semiparametrictransfer.data_sets.robonet_dataloader import FilteredRoboNetDataset
from semiparametrictransfer.models.invm_embed import GCBCImages

current_dir = os.path.dirname(os.path.realpath(__file__))

conf.configuration.update(AttrDict(
    model=GCBCImages,
    dataset_class=FilteredRoboNetDataset,
    data_dir= os.environ['DATA'] + '/misc_datasets' + '/robonet/robonet/hdf5'
    # data_dir = '/parent/media/febert/harddrive/misc_datasets/robonet/robonet_sampler/hdf5'
)
)

conf.model_config.update(AttrDict(
    # class_loss_mult=0
))

conf.data_config.update(AttrDict(
    robot_list=['widowx', 'sawyer', 'baxter'],
    # robot_list=['sawyer'],
    # train_val_split=[0.9, 0.05, 0.05],
    train_val_split=[0.7, 0.15, 0.15],
    color_augmentation=0.3,
    random_crop=True
))

configuration = conf.configuration
model_config = conf.model_config
data_config = conf.data_config

