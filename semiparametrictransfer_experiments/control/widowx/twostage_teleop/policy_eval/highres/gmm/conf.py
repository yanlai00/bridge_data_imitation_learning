from semiparametrictransfer_experiments.control.widowx.twostage_teleop.policy_eval.conf import config as baseconf
import os


config = baseconf

config['agent'].update({
    'image_height': 112,  # beceause of center crop
    'image_width': 144,
})

config['policy']['model_override_params'].update({
    'data_conf': {
        'data_dir': os.environ['DATA'] + '/spt_trainingdata/control/widowx/2stage_teleop/raw/large/stage0/clone/',
        'random_crop':[96, 128],
        'image_size_beforecrop':[112, 144]
    },
    'img_sz': [96, 128]
})

config['policy']['restore_path'] = '/mount/harddrive/experiments/spt_experiments/modeltraining/widowx/real/put_spoon_tray/bc_fromscratch/highres/gmm/weights/weights_itr190058.pth'