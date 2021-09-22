from doodad.wrappers.easy_launch import sweep_function, save_doodad_config
from semiparametrictransfer.train import ModelTrainer
import argparse
import os
import copy

def train(doodad_config, variant):
    args = argparse.Namespace()
    d = vars(args)
    for key, val in variant.items():
        d[key] = val
    ModelTrainer(args)

    save_doodad_config(doodad_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to the config file directory", required=True)
    parser.add_argument("--prefix", help="experiment prefix, if given creates subfolder in experiment directory", required=True)
    parser.add_argument("--dry", action='store_true', help="dry run, local no doodad")
    args = parser.parse_args()

    if not os.path.isabs(args.path):
        raise ValueError(' experiment path must be absolute!')

    sweep_par = {
        'data_config_override':
        # 'source_data_config_override':
        [
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/toysink3_bww/flip_cup_upright/lmdb'},
            #  'flip_cup_upright'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/toysink3_bww/put_brush_into_pot_or_pan/lmdb'},
            #  'put_brush_into_pot_or_pan'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/toysink3_bww/put_cup_into_pot_or_pan/lmdb'},
            #  'put_cup_into_pot_or_pan'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/toysink3_bww/put_detergent_from_sink_into_drying_rack/lmdb'},
            #  'put_detergent_from_sink_into_drying_rack'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/toysink3_bww/put_knife_in_pot_or_pan/lmdb'},
            #  'put_knife_in_pot_or_pan'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/toysink3_bww/take_lid_off_pot_or_pan/lmdb'},
            #  'take_lid_off_pot_or_pan'),

            ({'data_dir': '/robonetv2/toykitchen_fixed_cam/toysink3_bww/turn_lever_vertical_to_front/lmdb'},
             'turn_lever_vertical_to_front'),
            ({'data_dir': '/robonetv2/toykitchen_fixed_cam/toysink3_bww/put_green_squash_into_pot_or_pan/lmdb'},
             'put_green_squash_into_pot_or_pan'),
            ({'data_dir': '/robonetv2/toykitchen_fixed_cam/toysink3_bww/put_lid_on_pot_or_pan/lmdb'},
             'put_lid_on_pot_or_pan'),
            ({'data_dir': '/robonetv2/toykitchen_fixed_cam/toysink3_bww/put_pot_or_pan_from_sink_into_drying_rack/lmdb'},
             'put_pot_or_pan_from_sink_into_drying_rack'),

            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/put_carrot_in_pot_or_pan/lmdb'},
            #  'put_carrot_in_pot_or_pan'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/put_knife_on_cutting_board/lmdb'},
            #  'put_knife_on_cutting_board'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/put_potato_in_pot_or_pan/lmdb'},
            #  'put_potato_in_pot_or_pan'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/put_pot_or_pan_in_sink/lmdb'},
            #  'put_pot_or_pan_in_sink'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/put_pot_or_pan_on_stove/lmdb'},
            #  'put_pot_or_pan_on_stove'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/put_sweet_potato_in_pot/lmdb'},
            #  'put_sweet_potato_in_pot'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/toykitchen2_room8052/turn_lever_vertical_to_front/lmdb'},
            #  'turn_lever_vertical_to_front'),

            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/flip_pot_upright_in_sink_distractors/lmdb'}, 'flip_pot_upright'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/turn_faucet_front_to_left/lmdb'}, 'turn_faucet_front_to_left'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/put_corn_in_pan_which_is_on_stove_distractors/lmdb'}, 'put_corn_in_pan_which_is_on_stove'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/pick_up_pot_from_sink_distractors/lmdb'}, 'pick_up_pot_from_sink'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/put_corn_in_pot_which_is_in_sink_distractors/lmdb'}, 'put_corn_in_pot_which_is_in_sink'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/put_pot_on_stove_which_is_near_stove_distractors/lmdb'}, 'put_pot_on_stove_which_is_near_stove'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/pick_up_pan_from_stove_distractors/lmdb'}, 'pick_up_pan_from_stove'),
            # ##
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/put_knife_on_cutting_board/lmdb'}, 'put_knife_on_cutting_board'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/put_carrot_on_plate/lmdb'}, 'put_carrot_on_plate'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/put_lid_on_pot_or_pan/lmdb'}, 'put_lid_on_pot_or_pan'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/pick_up_pan_from_stove_distractors/lmdb'}, 'pick_up_pan_from_stove_distractors'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/put_pepper_in_pot_or_pan/lmdb'}, 'put_pepper_in_pot_or_pan'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/put_sweet_potato_in_pan_which_is_on_stove/lmdb'}, 'put_sweet_potato_in_pan_which_is_on_stove'),
            # ({'data_dir': '/robonetv2/toykitchen_fixed_cam/put_sweet_potato_in_pot_which_is_in_sink_distractors/lmdb'}, 'put_sweet_potato_in_pot_which_is_in_sink'),
            ]
    }


    default_params = {
        'gpu': -1,
        'strict_weight_loading': True,
        'deterministic': False,
        'cpu': False,
        'metric': False,
        'no_val': False,
        'no_train': False,
        'skip_main': False,
        'resume': None,
        'new_dir': True,
        'path': args.path,
        'prefix': args.prefix,
        'data_config_override': None,
        'load_task_indices': None,
        'source_data_config_override': None
    }

    for key in sweep_par:
        sweep_elements = sweep_par[key]
        for element in sweep_elements:
            params = copy.deepcopy(default_params)
            params[key] = element[0]
            params['prefix'] = str(args.prefix) + element[1]
            if args.dry:
                mode='here_no_doodad'
            else:
                mode='azure'
            sweep_function(
                train,
                {},
                default_params=params,
                log_path=str(args.prefix) + element[1],
                mode=mode,
                use_gpu=True,
                num_gpu=1,
            )



