from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer_experiments.modeltraining.widowx.real.viewpoint_invariance_pickup_3dof.stack_goal_image import conf
configuration = conf.configuration
data_config = conf.data_config
model_config = conf.model_config
data_config.main.dataconf.get_final_image_from_same_traj = False
data_config.main.dataconf.final_image_match = 'task'
data_config.main.dataconf.stack_goal_images = 8

val_keys = [k for k in data_config.main.keys() if k.startswith('val')]
for key in val_keys:
    data_config.main[key].dataconf.get_final_image_from_same_traj = False
    data_config.main[key].dataconf.final_image_match = 'task'
    data_config.main[key].dataconf.stack_goal_images = 8

model_config.main.stack_goal_images = 8
model_config.main.separate_classifier = True
model_config.main.separate_encoder = True
