from doodad.wrappers.easy_launch import sweep_function, save_doodad_config
from semiparametrictransfer.train import ModelTrainer
import argparse
import os

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
        raise ValueError('experiment path must be absolute!')

    params_to_sweep = {}
    default_params = {
        'gpu': -1,
        'strict_weight_loading': True,
        'deterministic': False,
        'cpu': False,
        'metric': False,
        'no_val': False,
        'no_train': False,
        'skip_main': False,
        'resume': '',
        'new_dir': True,
        'path': args.path,
        'prefix': args.prefix,
        'data_config_override': None,
        'source_data_config_override': None,
        'load_task_indices': None,
    }
    if args.dry:
        mode = 'here_no_doodad'
        use_gpu = True
    else:
        mode = 'azure'
        use_gpu = True
    sweep_function(
        train,
        params_to_sweep,
        default_params=default_params,
        log_path=args.prefix,
        mode=mode,
        use_gpu=use_gpu,
        num_gpu=1,
    )


