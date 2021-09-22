from semiparametrictransfer_experiments.modeltraining.widowx.real.toy_kitchen_v0.bc_fromscratch import conf
import os
configuration = conf.configuration
data_config = conf.data_config
model_config = conf.model_config

data_config.main.dataconf.data_dir = [
            os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/put_corn_in_pan_which_is_on_stove_distractors/lmdb',
            ]
