from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer_experiments.modeltraining.widowx.real.viewpoint_invariance_pickup_3dof.gcbc import conf
configuration = conf.configuration
data_config = conf.data_config
model_config = conf.model_config
data_config.main.dataconf.get_final_image_from_same_traj = False
data_config.main.dataconf.final_image_match = 'task'
model_config.main.separate_classifier = True
configuration.batch_size = 16
