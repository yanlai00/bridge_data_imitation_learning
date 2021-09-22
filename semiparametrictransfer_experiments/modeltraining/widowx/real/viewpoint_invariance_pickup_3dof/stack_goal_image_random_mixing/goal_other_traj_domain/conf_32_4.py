from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer_experiments.modeltraining.widowx.real.viewpoint_invariance_pickup_3dof.stack_goal_image_random_mixing import conf
from semiparametrictransfer_experiments.modeltraining.widowx.real.viewpoint_invariance_pickup_3dof.dataset_lmdb import bridge_data_test
configuration = conf.configuration
data_config = conf.data_config
model_config = conf.model_config

train_keys = [k for k in data_config.main.dataconf.keys()]
for train_key in train_keys:
    data_config.main.dataconf[train_key][1].get_final_image_from_same_traj = False
    data_config.main.dataconf[train_key][1].final_image_match = 'domain'
    data_config.main.dataconf[train_key][1].stack_goal_images = 4

val_keys = [k for k in data_config.main.keys() if k.startswith('val')]
for key in val_keys:
    data_config.main[key].dataconf.get_final_image_from_same_traj = False
    data_config.main[key].dataconf.final_image_match = 'domain'
    data_config.main[key].dataconf.stack_goal_images = 4

model_config.main.stack_goal_images = 4
model_config.main.separate_classifier = True
