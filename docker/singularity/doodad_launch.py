from doodad.launch_tools import launch_python
import doodad.mount as mount

from doodad.mode import SlurmSingularity
from doodad.slurm.slurm_util import SlurmConfig

my_slurm_config = SlurmConfig(
            account_name='co_rail',
            partition='savio3_2080ti',
            time_in_mins=60*30,
            max_num_cores_per_node=16,
            n_gpus=1,
            n_cpus_per_task=2,
            n_nodes=1,
        )

def create_mode():

    # Change the image and set the GPU mode accordingly
    return SlurmSingularity(
        image='/global/home/groups/co_rail/febert/singularity_images/singularity_bare_gpu.sif',
        gpu=True,
        slurm_config=my_slurm_config
    )


def create_mounts():
    return [
        # Point to your code.
        # So if you have "import code" or "import code2", it
        # should look something like
        mount.MountLocal(
            local_dir='/global/home/users/febert/code/semiparametrictransfer', pythonpath=True,
        ),

        # Point to non-code directories, like mujoco. Note the lack of "pythonpath".
        mount.MountLocal(
            local_dir='/global/home/USERNAME/.mujoco',
            mount_point='/root/.mujoco',
        ),

        # Your script should output to this directory directly, unlike other modes
        # like that have separate "mount_points".
        mount.MountLocal(local_dir='/global/scratch/febert/experiments/spt_experiments',
                         mount_point='/global/scratch/febert/experiments/spt_experiments',
                         output=True,
                         ),
    ]

if __name__ == '__main__':
    launch_python(
        target='/global/home/users/febert/code/semiparametrictransfer/semiparametrictransfer/train.py',
        args={'':'experiments/modeltraining/sawyer/bridge_targetfinetune/conf.py'},
        mode=create_mode(),
        mount_points=create_mounts(),
    )

