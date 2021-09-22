import os.path
BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.policies.gcbc_policy import GCBCPolicyImages
from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
from widowx_envs.widowx.widowx_env import WidowXEnv
from widowx_envs.control_loops import TimedLoop

env_params = {
    # 'camera_topics': [IMTopic('/cam1/image_raw')],
    'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw'), IMTopic('/cam2/image_raw'), IMTopic('/cam3/image_raw'), IMTopic('/cam4/image_raw')],
    'gripper_attached': 'custom',
    'skip_move_to_neutral': True,
    'action_mode':'3trans',
    'fix_zangle': 0.1,
    'move_duration': 0.2,
    'override_workspace_boundaries': [[0.2, -0.04, 0.03, -1.57, 0], [0.31, 0.04, 0.1,  1.57, 0]]
}

agent = {
    'type': TimedLoop,
    'env': (WidowXEnv, env_params),
    'T': 30,
    'image_height': 56,
    'image_width': 72,
    'make_final_gif': False,
    # 'video_format': 'gif',   # already by default
    'recreate_env': (False, 1),
    'ask_confirmation': False,
    'stack_goal_images': 4,
    # 'load_goal_image': [os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww_printer_room/pick_blue_elephant/2021-05-29_20-50-42/raw/traj_group0/traj0', 39],
    # 'load_goal_image': [os.environ['DATA'] + '/robonetv2/vr_record_applied_actions_robonetv2/bww_printer_room/test_goal_image/list_cow/2021-05-29_21-43-20/raw/traj_group0/traj0', 16],
    'load_goal_image': '/home/datacol1/robonetv2bucket/spt_data/trainingdata/robonetv2/vr_record_applied_actions_robonetv2/lmdb_all/pick_green_frog/lmdb',
}

policy = {
    'type': GCBCPolicyImages,
    # BC fromscratch
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/viewpoint_invariance_pickup_3dof/bc_fromscratch/camera_class/randcam/weights/weights_itr199916.pth',

    #GCBC
    # 'restore_path': os.environ['EXP'] + '/brc/lmdb_real_world_05-03-21/real/viewpoint_invariance_pickup_3dof/gcbc/camera_class/add_single_sourcedata_to_bridge_cl0.01/weights/weights_itr198704.pth',
    # 'restore_path': os.environ['EXP'] + '/brc/lmdb_real_world_05-03-21/real/viewpoint_invariance_pickup_3dof/gcbc/add_single_sourcedata_to_bridge/weights/weights_itr198704.pth',
    #GCBC with more backgrounds:
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/viewpoint_invariance_pickup_3dof/gcbc/camera_class/multi_backgr/weights/weights_itr398596.p?th',
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/viewpoint_invariance_pickup_3dof/gcbc/multi_backgr/weights/weights_itr398596.pth',

    #GCBC image_level task conditioning:
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/viewpoint_invariance_pickup_3dof/gcbc_imagelevel_taskconditioning/weights/weights_itr217416.pth',
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/viewpoint_invariance_pickup_3dof/gcbc_imagelevel_taskconditioning/camera_class/weights/weights_itr217416.pth',
    # 'pass_zero_goal_image':True,

    #Pretraining and Finetuning
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/viewpoint_invariance_pickup_3dof/bridge_targetfinetune/camera_class/single_source_vp/finetuning/weights/weights_itr142074.pth',
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/viewpoint_invariance_pickup_3dof/bridge_targetfinetune/single_source_vp/finetuning/weights/weights_itr142074.pth',

    # GCBC with goal-swapping 4.5k traj
    # 'restore_path': os.environ['AZURE_EXP'] + '/spt_experiments/modeltraining/widowx/real/viewpoint_invariance_pickup_3dof/gcbc/goal_other_traj/real_gcbc_goal_other_traj_0528_2021-05-29_06-22-20/weights/weights_itr50001.pth',
    # GCBC  4.5k traj
    # 'restore_path': os.environ['AZURE_EXP'] + '/spt_experiments/modeltraining/widowx/real/viewpoint_invariance_pickup_3dof/gcbc/real_gcbc_0529_2021-05-29_08-52-26/weights/weights_itr185773.pth',
    # classifier
    # 'restore_path': os.environ['AZURE_EXP'] + '/spt_experiments/modeltraining/widowx/real/viewpoint_invariance_pickup_3dof/gcbc/camera_class/real_gcbc_classifier_0529_2021-05-29_08-03-37/weights/weights_itr185773.pth',

    # task-id conditioned (random mixing 30-70)

    # stack goal images matching domain separate encoder 8 images
    # 'restore_path': os.environ['AZURE_EXP'] + '/spt_experiments/modeltraining/widowx/real/viewpoint_invariance_pickup_3dof/stack_goal_image/goal_other_traj_domain_sep_encoder/stack8_domain_sep_encoder_0604_2021-06-04_09-58-02/weights/weights_best_itr132695.pth',

    # stack goal images matching domain shared encoder 4 images
    'restore_path': os.environ['AZURE_EXP'] + '/spt_experiments/modeltraining/widowx/real/viewpoint_invariance_pickup_3dof/stack_goal_image/goal_other_traj_domain/stack4_b32_domain_0604_2021-06-04_07-43-19/weights/weights_best_itr106156.pth',

    # stack goal images matching domain separate encoder 4 images
    # 'restore_path': os.environ['AZURE_EXP'] + '/spt_experiments/modeltraining/widowx/real/viewpoint_invariance_pickup_3dof/stack_goal_image/goal_other_traj_domain_sep_encoder/stack4_domain_sep_encoder_0604_2021-06-04_09-56-50/weights/weights_best_itr53078.pth',

    # stack goal images matching task separate encoder 8 images
    # 'restore_path': os.environ['AZURE_EXP'] + '/spt_experiments/modeltraining/widowx/real/viewpoint_invariance_pickup_3dof/stack_goal_image/goal_other_traj_task_sep_encoder/stack8_task_sep_encoder_0604_2021-06-04_09-42-54/weights/weights_best_itr132695.pth',

    # stack goal images matching task shared encoder 8 images
    # 'restore_path': os.environ['AZURE_EXP'] + '/spt_experiments/modeltraining/widowx/real/viewpoint_invariance_pickup_3dof/stack_goal_image/goal_other_traj_task/stack8_b32_task_0604_2021-06-04_07-40-26/weights/weights_best_itr132695.pth',

    # stack goal images matching task shared encoder 4 images
    # 'restore_path': os.environ['AZURE_EXP'] + '/spt_experiments/modeltraining/widowx/real/viewpoint_invariance_pickup_3dof/stack_goal_image/goal_other_traj_task/stack4_b32_task_0604_2021-06-04_07-41-51/weights/weights_best_itr79617.pth',

    'confirm_first_image': True,
    'stack_goal_images': True,
    'model_override_params': {
        'data_conf': {
            'random_crop': [48, 64],
            'image_size_beforecrop': [56, 72],
        },
        'img_sz': [48, 64],
        'sel_camera': 3,
        # 'test_time_task_id': 1
    }
}

config = {
    # 'collection_metadata' : current_dir + '/collection_metadata.json',
    'current_dir' : current_dir,
    'start_index':0,
    'end_index': 300,
    'agent': agent,
    'policy': policy,
    # 'save_data': True,  # by default
    'save_format': ['raw'],
}