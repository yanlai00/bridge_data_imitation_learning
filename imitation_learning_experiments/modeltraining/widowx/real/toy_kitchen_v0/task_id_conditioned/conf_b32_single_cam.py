from semiparametrictransfer_experiments.modeltraining.widowx.real.toy_kitchen_v0.task_id_conditioned import conf_b32 as conf
configuration = conf.configuration
data_config = conf.data_config
model_config = conf.model_config

def filtering_cam0(data_frame):
    return data_frame[(data_frame['camera_index'] == 0)]

data_config.main.dataconf.filtering_function = [filtering_cam0]