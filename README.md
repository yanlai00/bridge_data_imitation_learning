# Installation

In your `.bashrc` set the environment variables EXP for experiment data and DATA for trainingdata:

```
export EXP='/mount/harddrive/experiments'
export DATA='/mount/harddrive/trainingdata'
```

Setup conda environment by running
`conda env create -f py3_environment.yml`

then in this directory run

`python setup develop`

Clone git@github.com:rail-berkeley/private_visual_foresight.git and checkout dev_frederik
run `git submodule update --init --recursive` to load the submodules.
then run

`python setup develop`



Use the master branch of git@github.com:rail-berkeley/RoboNet-private.git
then run

`python setup develop` 



# Examples to run

## Data collection
To collect the source data (picking gatorade bottle and holding it over the bowl) run:
`python imitation_learning/run_control_experiment.py experiments/control/widowx/sim/pick_only_gatorade/conf.py`

Collect data for lifting objects over a bowl with a random viewpoint, selecting 2 out of 5 objects randomly:
`python imitation_learning/run_control_experiment.py experiments/control/widowx/sim/randview_2out5obj_nogato/conf.py`


## Training
To train on the source task only run:
`python imitation_learning/train.py --path  imitation_learning_experiments/bc_fromscratch/conf.py`

To jointly train on bridge and source data run:
`python imitation_learning/train.py --path  imitation_learning_experiments/task_id_conditioning/conf.py`

To add the domain adverserial loss run:
`python imitation_learning/train.py --path  imitation_learning_experiments/random_mixing_task_id/conf.py`
