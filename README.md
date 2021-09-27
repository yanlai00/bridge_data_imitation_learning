# Bridge Data Imitation Learning

This is the accompanying code repository for the paper "Bridge Data: Boosting Generalization of Robotic Skills with Cross-Domain Datasets".

## Installation

In your `.bashrc` set the environment variables EXP for experiment data and DATA for trainingdata:

```
export EXP=<folder to store experiment results>
export DATA=<folder where the bridge dataset is stored>
```

Setup conda environment by running

```
conda create --name bridgedata python=3.6.8 pip
pip install -r requirements.txt
```

then in this directory run

`python setup.py develop`

Clone the  [bridge data robot infrastructure repository](https://github.com/yanlai00/bridge_data_robot_infra), install the dependencies, and run

`python widowx_envs/setup.py develop`

## Examples to run

### Training

#### Single Task Imitation Learning

`python bridgedata/train.py --path  bridgedata_experiments/bc_fromscratch/conf.py`

The example config file trains the "wipe plate with sponge task". You can change the training task and the training parameters in `bridgedata_experiments/bc_fromscratch/conf.py`.

#### Multi Task Imitation Learning

`python bridgedata/train.py --path  bridgedata_experiments/task_id_conditioning/conf.py`

The example config file trains a multi-task, task-id conditioned imitation learning policy on all of the tasks in toykitchen1.  

Another example config file `bridgedata_experiments/task_id_conditioning/conf_exclude_toykitchen1.py` trains a multi-task policy on all of the environments except toykitchen1 (to evaluation transferability of policies).

#### Multi Task Imitation Learning (with dataset re-balancing)

`python bridgedata/train.py --path  bridgedata_experiments/random_mixing_task_id/conf.py`

The example config file trains a multi-task, task-id conditioned imitation learning policy on all of the environments except real kitchen 1, and the wipe plate with sponge task. The dataset is re-balanced such that the wipe plate with sponge task takes up 10% of the training dataset.  

Another example config file `bridgedata_experiments/random_mixing_task_id/conf_toykitchen1.py` rebalances the dataset such that trajectories in toy kitchen 1 takes up 30% of the training dataset.

## Doodad

This repository also provides an example script  `docker/azure/doodad_launch.py` for launching jobs on cloud compute services like AWS, GCP or Azure with [Doodad](https://github.com/rail-berkeley/doodad).
