# Installation

In your `.bashrc` set the environment variables EXP for experiment data and DATA for trainingdata:

```
export EXP=<folder to store experiment results>
export DATA=<folder where the bridge dataset is stored>
```

Setup conda environment by running
`conda create --name bridgedata python=3.6.8 pip`
`pip install -r requirements.txt`

then in this directory run

`python setup.py develop`

Clone the bridge data robot infrastructure repository and run

`python setup.py develop`

# Examples to run

## Training
To train on a single task only run:
`python imitation_learning/train.py --path  imitation_learning_experiments/bc_fromscratch/conf.py`

To jointly train on multiple tasks in the bridge dataset with task-id conditioning, run:
`python imitation_learning/train.py --path  imitation_learning_experiments/task_id_conditioning/conf.py`

To jointly train on multiple tasks in the bridge dataset with dataset re-balancing, run:
`python imitation_learning/train.py --path  imitation_learning_experiments/random_mixing_task_id/conf.py`
