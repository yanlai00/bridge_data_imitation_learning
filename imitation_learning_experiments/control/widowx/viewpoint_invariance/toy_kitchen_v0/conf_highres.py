import os.path
BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))
from semiparametrictransfer.policies.gcbc_policy import GCBCPolicyImages
from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic
from widowx_envs.widowx.widowx_env import WidowXEnv
from widowx_envs.control_loops import TimedLoop

# load_traj = os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/flip_pot_upright_in_sink_distractors/2021-06-02_17-08-30/raw/traj_group0/traj17'
# load_traj = os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/put_corn_in_pot_which_is_in_sink_distractors/2021-06-04_16-31-05/raw/traj_group0/traj16' # traj 8 and traj 16
# load_traj = os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/put_sweet_potato_in_pot_which_is_in_sink_distractors/2021-06-03_16-50-31/raw/traj_group0/traj21' # traj 17 and traj 21
# load_traj = os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/put_sweet_potato_in_pan_which_is_on_stove_distractors/2021-06-03_19-00-26/raw/traj_group0/traj27'
# load_traj = os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/put_carrot_on_plate/2021-06-06_17-56-18/raw/traj_group0/traj10' # first folder traj 10 and second folder traj 9
# load_traj = os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/put_knife_on_cutting_board/2021-06-06_19-35-44/raw/traj_group0/traj11' # traj 3 and 15 and 11
# load_traj = os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/put_corn_in_pan_which_is_on_stove_distractors/2021-06-03_17-40-44/raw/traj_group0/traj9' # traj 6 and traj 9
# load_traj = os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/pick_up_pan_from_stove_distractors/2021-06-03_18-53-07/raw/traj_group0/traj2'
# load_traj = os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/put_lid_on_pot_or_pan/2021-06-06_17-14-25/raw/traj_group0/traj10' # traj 10
# load_traj = os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/put_pepper_in_pot_or_pan/2021-06-06_19-51-49/raw/traj_group0/traj6'
# load_traj = os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/put_eggplant_on_plate/2021-06-08_17-57-18/raw/traj_group0/traj2'
# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toysink1_room8052/put_egglant_into_pan/2021-06-10_19-42-55/raw/traj_group0/traj8', 3]
# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/initial_test_config/2021-06-15_15-48-23/raw/traj_group0/traj0', 15]
# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/turn_lever_vertical_to_front/2021-06-11_19-51-22/raw/traj_group0/traj15', 0]
# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/put_pot_or_pan_in_sink/2021-06-11_15-30-06/raw/traj_group0/traj0', 4]
# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/put_pot_or_pan_on_stove/2021-06-11_16-14-25/raw/traj_group0/traj6', 5]
# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/lift_bowl/2021-06-16_13-30-23/raw/traj_group0/traj8', 0]
# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/lift_bowl/2021-06-16_13-07-28/raw/traj_group0/traj18', 0]
# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/put_pear_in_bowl/2021-06-16_14-35-31/raw/traj_group0/traj2', 0]
# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/put_pear_in_bowl/2021-06-16_14-59-12/raw/traj_group0/traj1', 0]
# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/flip_orange_pot_upright_in_sink/2021-06-16_13-55-15/raw/traj_group0/traj0', 0]
# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/flip_orange_pot_upright_in_sink/2021-06-16_14-13-45/raw/traj_group0/traj11', 0]
# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/initial_test_config/2021-06-17_16-21-23/raw/traj_group0/traj0', 90]
# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/put_pot_or_pan_in_sink/2021-06-11_15-45-56/raw/traj_group0/traj0', 0]
# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam_initial_testconfig/initial_test_config/2021-06-22_13-37-06/raw/traj_group0/traj0', 75]
load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/put_potato_on_plate/2021-07-05_15-33-28/raw/traj_group0/traj8', 0]
# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/put_can_in_pot/2021-07-05_18-02-02/raw/traj_group0/traj8', 0]
# load_traj = [os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/put_corn_on_plate/2021-07-05_20-30-06/raw/traj_group0/traj20', 0]

env_params = {
    # 'camera_topics': [IMTopic('/cam1/image_raw')],
    'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw'), IMTopic('/cam2/image_raw')],
    'gripper_attached': 'custom',
    'skip_move_to_neutral': True,
    # 'action_mode':'3trans3rot',
    'fix_zangle': 0.1,
    'move_duration': 0.2,
    'override_workspace_boundaries': [[0.2, -0.04, 0.03, -1.57, 0], [0.31, 0.04, 0.1,  1.57, 0]],
    'action_clipping': None,
    # 'start_transform': os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/initial_testconfig/2021-06-02_13-52-50/raw/traj_group0/traj0',
    # 'start_transform': os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/test/2021-06-05_21-11-00/raw/traj_group0/traj0',
    # 'start_transform': os.environ['DATA'] + '/robonetv2/toykitchen_fixed_cam/flip_pot_upright_in_sink_distractors/2021-06-02_16-53-06/raw/traj_group0/traj0/'
    'start_transform': load_traj,
    # cropped, success: 0, 3, 53
    # whole_large: fail 0, 3,  success 53
    # whole_small: fail 0, 3, 53, 5
}

agent = {
    'type': TimedLoop,
    'env': (WidowXEnv, env_params),
    'T': 50,
    'image_height': 480,  # for highres
    'image_width': 640,   # for highres
    'make_final_gif': False,
    # 'video_format': 'gif',   # already by default
    'recreate_env': (False, 1),
    'ask_confirmation': False,
    # 'load_goal_image': [load_traj, 18],
}

from widowx_envs.policies.vr_teleop_policy import VRTeleopPolicy
from widowx_envs.policies.policy import NullPolicy
policy = [
{
    'type': GCBCPolicyImages,

    # task-id with toykitchen2 only
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/task_id_conditioned/b32_toykitchen2_2021-06-12_19-55-09/weights/weights_itr400001.pth',
        # "human_demo, put potato in pot or pan": 0,
        # "human_demo, put pot or pan in sink": 1,
        # "human_demo, put pot or pan on stove": 2,
        # "human_demo, turn lever vertical to front": 3,
        # "human_demo, put knife on cutting board": 4,
        # "human_demo, put sweet potato in pot": 5,
        # "human_demo, put carrot in pot or pan": 6

    # toykitchen 2 all data random mixing aliasing
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/random_mixing_task_id/b32_alldata_random_mixing_aliasing_2021-06-16_23-11-29/weights/weights_best_itr67545.pth',
    #     "human_demo, take lid off pot or pan": 0,
    #     "human_demo, put lid on stove": 1,
    #     "human_demo, put spoon in pot": 2,
    #     "human_demo, put carrot in pot or pan": 3,
    #     "human_demo, put knife in pot or pan": 4,
    #     "human_demo, put corn in pot which is in sink": 5,
    #     "human_demo, put knife on cutting board": 6,
    #     "human_demo, put eggplant on plate": 7,
    #     "human_demo, pick up pot from sink": 8,
    #     "human_demo, twist knob start vertical _clockwise90": 9,
    #     "human_demo, flip pot upright which is in sink": 10,
    #     "human_demo, put cup into pot or pan": 11,
    #     "human_demo, put sweet_potato in pan which is on stove": 12,
    #     "human_demo, pick up pan from stove": 13,
    #     "human_demo, put eggplant into pot or pan": 14,
    #     "human_demo, turn lever vertical to front": 15,
    #     "human_demo, put carrot on plate": 16,
    #     "human_demo, put spoon into pan": 17,
    #     "human_demo, put cup from anywhere into sink": 18,
    #     "human_demo, put green squash into pot or pan": 19,
    #     "human_demo, put carrot on cutting board": 20,
    #     "human_demo, put potato in pot or pan": 21,
    #     "human_demo, put pot or pan from sink into drying rack": 22,
    #     "human_demo, put pot or pan on stove": 23,
    #     "human_demo, turn faucet front to left (in the eyes of the robot)": 24,
    #     "human_demo, put pepper in pot or pan": 25,
    #     "human_demo, flip cup upright": 26,
    #     "human_demo, put corn in pan which is on stove": 27,
    #     "human_demo, put brush into pot or pan": 28,
    #     "human_demo, put sweet potato in pot": 29,
    #     "human_demo, put pot or pan in sink": 30,
    #     "human_demo, put lid on pot or pan": 31,
    #     "human_demo, put detergent from sink into drying rack": 32



    # task-id excluding kitchen2 (direct transfer)
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/task_id_conditioned/b32_excl_kit2_2021-06-15_06-07-02/weights/weights_best_itr52232.pth',
        # "human_demo, put spoon into pan": 0,
        # "human_demo, put pan from sink into drying rack": 1,
        # "human_demo, put pepper in pot or pan": 2,
        # "human_demo, put cup into pot or pan": 3,
        # "human_demo, twist knob start vertical _clockwise90": 4,
        # "human_demo, flip cup upright": 5,
        # "human_demo, put corn in pan which is on stove": 6,
        # "human_demo, put pot in sink": 7,
        # "human_demo, put eggplant on plate": 8,
        # "human_demo, put detergent from sink into drying rack": 9,
        # "human_demo, put pot or pan on stove": 10,
        # "human_demo, put eggplant into pot or pan": 11,
        # "human_demo, put pan on stove from sink": 12,
        # "human_demo, take lid off pot or pan": 13,
        # "human_demo, pick up pot from sink": 14,
        # "human_demo, put lid on pot or pan": 15,
        # "human_demo, put pan from stove to sink": 16,
        # "human_demo, turn lever vertical to front": 17,
        # "human_demo, put green squash in pot or pan ": 18,
        # "human_demo, put sweet_potato in pan which is on stove": 19,
        # "human_demo, put pot or pan from sink into drying rack": 20,
        # "human_demo, flip pot upright which is in sink": 21,
        # "human_demo, put carrot on cutting board": 22,
        # "human_demo, put sweet_potato in pot which is in sink": 23,
        # "human_demo, put lid on stove": 24,
        # "human_demo, put knife on cutting board": 25,
        # "human_demo, put pan from drying rack into sink": 26,
        # "human_demo, put spoon in pot": 27,
        # "human_demo, turn faucet front to left (in the eyes of the robot)": 28,
        # "human_demo, put carrot in pot or pan": 29,
        # "human_demo, put green squash into pot or pan": 30,
        # "human_demo, put eggplant in pot or pan": 31,
        # "human_demo, put pot or pan in sink": 32,
        # "human_demo, put potato in pot or pan": 33,
        # "human_demo, put sweet potato in pot": 34,
        # "human_demo, put pan in sink": 35,
        # "human_demo, put corn in pot which is in sink": 36,
        # "human_demo, put cup from anywhere into sink": 37,
        # "human_demo, pick up pan from stove": 38,
        # "human_demo, put brush into pot or pan": 39,
        # "human_demo, put eggplant into pan": 40,
        # "human_demo, put cup from counter or drying rack into sink": 41,
        # "human_demo, put knife in pot or pan": 42,
        # "human_demo, put carrot on plate": 43,
        # "human_demo, put pot on stove which is near stove": 44

    # Single task BC for toykitchen2
    # sweet potato
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/bc_fromscratch/sweep_singletask_put_sweet_potato_in_pot_2021-06-14_06-35-05/weights/weights_best_itr40455.pth',
    # potato
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/bc_fromscratch/sweep_singletask_put_potato_in_pot_or_pan_2021-06-14_06-24-36/weights/weights_best_itr40455.pth',
    # put pot or pan in sink
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/bc_fromscratch/singletask_pot_in_sink_2021-06-15_06-11-42/weights/weights_best_itr111028.pth',

    # transfer from target domain to new domain (toykitchen 2 to toysink 1 on the potato task)
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/task_id_conditioned/b32_alldata_2021-06-15_06-06-08/weights/weights_best_itr105070.pth',
    # "human_demo, put potato in pot or pan": 11,

    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/random_mixing_task_id/b32_rndmix_2021-06-15_06-10-51//weights/weights_best_itr82555.pth',
    # "human_demo, put potato in pot or pan": 9,

    # toykitchen1, toykitchen2, and toysink2 (for direct transfer to toysink1)
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/task_id_conditioned/b32_excl_sin1_2021-06-14_05-41-54/weights/weights_best_itr54050.pth',
        # "human_demo, put pan on stove from sink": 0,
        # "human_demo, twist knob start vertical _clockwise90": 1,
        # "human_demo, put pot or pan in sink": 2,
        # "human_demo, put pan from drying rack into sink": 3,
        # "human_demo, put carrot on cutting board": 4,
        # "human_demo, put eggplant in pot or pan": 5,
        # "human_demo, put spoon in pot": 6,
        # "human_demo, put potato in pot or pan": 7,
        # "human_demo, pick up pan from stove": 8,
        # "human_demo, put corn in pot which is in sink": 9,
        # "human_demo, put green squash in pot or pan ": 10,
        # "human_demo, put sweet_potato in pan which is on stove": 11,
        # "human_demo, flip pot upright which is in sink": 12,
        # "human_demo, put pepper in pot or pan": 13,
        # "human_demo, put pan in sink": 14,
        # "human_demo, put pot on stove which is near stove": 15,
        # "human_demo, put spoon into pan": 16,
        # "human_demo, put eggplant on plate": 17,
        # "human_demo, put eggplant into pot or pan": 18,
        # "human_demo, pick up pot from sink": 19,
        # "human_demo, turn lever vertical to front": 20,
        # "human_demo, put carrot in pot or pan": 21,
        # "human_demo, put sweet potato in pot": 22,
        # "human_demo, turn faucet front to left (in the eyes of the robot)": 23,
        # "human_demo, put pot in sink": 24,
        # "human_demo, put eggplant into pan": 25,
        # "human_demo, put carrot on plate": 26,
        # "human_demo, put pot or pan on stove": 27,
        # "human_demo, put lid on stove": 28,
        # "human_demo, put pan from sink into drying rack": 29,
        # "human_demo, put cup from counter or drying rack into sink": 30,
        # "human_demo, put knife on cutting board": 31,
        # "human_demo, put lid on pot or pan": 32,
        # "human_demo, put pan from stove to sink": 33,
        # "human_demo, put sweet_potato in pot which is in sink": 34,
        # "human_demo, put corn in pan which is on stove": 35

    # toysink 1 random mix (with toy kitchen 1)
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/random_mixing_task_id/toysink_randommix_2021-06-11_21-14-34/weights/weights_best_itr385599.pth',
        # "human_demo, put green squash in pot or pan ": 0,
        # "human_demo, twist knob start vertical _clockwise90": 1,
        # "human_demo, put sweet_potato in pan which is on stove": 2,
        # "human_demo, put carrot on cutting board": 3,
        # "human_demo, put pan from stove to sink": 4,
        # "human_demo, put eggplant in pot or pan": 5,
        # "human_demo, put sweet_potato in pot which is in sink": 6,
        # "human_demo, put corn in pot which is in sink": 7,
        # "human_demo, put lid on pot or pan": 8,
        # "human_demo, put pan from drying rack into sink": 9,
        # "human_demo, put corn in pan which is on stove": 10,
        # "human_demo, flip pot upright which is in sink": 11,
        # "human_demo, put pot on stove which is near stove": 12,
        # "human_demo, put lid on stove": 13,
        # "human_demo, put pot in sink": 14,
        # "human_demo, pick up pot from sink": 15,
        # "human_demo, put pan from sink into drying rack": 16,
        # "human_demo, put pan on stove from sink": 17,
        # "human_demo, pick up pan from stove": 18,
        # "human_demo, put eggplant on plate": 19,
        # "human_demo, put carrot on plate": 20,
        # "human_demo, turn faucet front to left (in the eyes of the robot)": 21,
        # "human_demo, put eggplant into pan": 22,
        # "human_demo, turn lever vertical to front": 23,
        # "human_demo, put pan in sink": 24,
        # "human_demo, put pepper in pot or pan": 25,
        # "human_demo, put spoon into pan": 26,
        # "human_demo, put knife on cutting board": 27

    # New tasks random mixing
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/random_mixing_task_id/b32_kitchen2_lift_bowl_2021-06-17_19-50-16/weights/weights_best_itr75400.pth',
    # "human_demo, lift bowl": 0,

    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/random_mixing_task_id/b32_kitchen2_put_pear_2021-06-17_20-03-29/weights/weights_best_itr98020.pth',
    # "human_demo, put pear in bowl": 4,

    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/random_mixing_task_id/b32_kitchen2_orange_pot_2021-06-17_20-01-25/weights/weights_best_itr75400.pth',
    # "human_demo, flip orange pot upright in sink": 24,

    # New tasks single task baseline
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/bc_fromscratch/put_pear_baseline_2021-06-18_02-44-49/weights/weights_best_itr133950.pth',
    # put pear

    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/bc_fromscratch/flip_orange_pot_baseline_2021-06-18_02-41-37/weights/weights_best_itr42333.pth',
    # flip ornage pot

    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/bc_fromscratch/lift_bowl_baseline_2021-06-18_04-24-31/weights/weights_best_itr74784.pth',
    # lift bowl

    # Single task BC for long trajectories
    # potato on plate
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/bc_fromscratch/potato_on_plate_single_task_0707_2021-07-08_02-07-09/weights/weights_best_itr44160.pth',
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/bc_fromscratch/potato_on_plate_single_task_0707_2021-07-08_02-07-09/weights/weights_itr200001.pth',

    # long traj task id conditioned
    # "human_demo, put potato on plate": 0,
    # "human_demo, put lemon on plate": 1,
    # "human_demo, put can in pot": 2,
    # "human_demo, put corn on plate": 3
    'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/task_id_conditioned/long_traj_only_2021-07-08_18-39-18/weights/weights_best_itr97700.pth',

    # random mixing with long traj
    # "human_demo, put spoon in pot": 0,
    # "human_demo, put lemon on plate": 1,
    # "human_demo, put carrot on plate": 2,
    # "human_demo, put pot or pan on stove": 3,
    # "human_demo, put green squash into pot or pan": 4,
    # "human_demo, put corn on plate": 5,
    # "human_demo, take lid off pot or pan": 6,
    # "human_demo, put knife on cutting board": 7,
    # "human_demo, turn lever vertical to front": 8,
    # "human_demo, flip salt upright": 9,
    # "human_demo, put eggplant into pot or pan": 10,
    # "human_demo, put cup into pot or pan": 11,
    # "human_demo, put sushi on plate": 12,
    # "human_demo, put spatula in pan": 13,
    # "human_demo, put strawberry in pot": 14,
    # "human_demo, put can in pot": 15,
    # "human_demo, put lid on pot or pan": 16,
    # "human_demo, put pot or pan in sink": 17,
    # "human_demo, put potato in pot or pan": 18,
    # "human_demo, put brush into pot or pan": 19,
    # "human_demo, put knife in pot or pan": 20,
    # "human_demo, put potato on plate": 21,
    # "human_demo, put detergent from sink into drying rack": 22,
    # "human_demo, flip cup upright": 23,
    # "human_demo, flip pot upright which is in sink": 24,
    # "human_demo, put sweet potato in pot": 25,
    # "human_demo, put carrot in pot or pan": 26,
    # "human_demo, flip orange pot upright in sink": 27,
    # "human_demo, put pear in bowl": 28,
    # "human_demo, lift bowl": 29,
    # "human_demo, put spoon into pan": 30,
    # "human_demo, put pot or pan from sink into drying rack": 31,
    # "human_demo, put cup from anywhere into sink": 32
    # 'restore_path': os.environ['EXP'] + '/spt_experiments/modeltraining/widowx/real/toy_kitchen_v0/random_mixing_task_id/long_traj_random_mixing_2021-07-08_18-41-20/weights/weights_best_itr317892.pth',

    'confirm_first_image': False,
    # 'crop_image_region': [31, 88],
    'model_override_params': {
        'data_conf': {
            'random_crop': [96, 128],
            'image_size_beforecrop': [112, 144]
        },
        'img_sz': [96, 128],
        'sel_camera': 0,
        'state_dim': 7,
        'test_time_task_id': 0,
    },
}
]


config = {
    # 'collection_metadata' : current_dir + '/collection_metadata.json',
    'current_dir': current_dir,
    'start_index': 0,
    'end_index': 300,
    'agent': agent,
    'policy': policy,
    'save_data': False,  # by default
    'save_format': ['raw'],
}