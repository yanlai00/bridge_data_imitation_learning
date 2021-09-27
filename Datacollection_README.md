# Data collection instructions

open termainal
ctrl alt t

to cancel program:
ctrl c

cd <dir>

to copy from terminal
ctrl shit c


to open directory in file browser:
ctrl l 


## robot script:
```(base) datacol@fermat:~/interbotix_ws/src/private_visual_foresight/visual_mpc/foresight_rospkg/launch$ ./start_interbotix_sdk.sh```

## camera launch script:
```(spt) datacol@fermat:~/interbotix_ws/src/private_visual_foresight/visual_mpc/foresight_rospkg/launch$ python start_cameras.py --use_connector_chartselect timeout```

To check if usb cameras are recognized run:
`$v4l2-ctl --list-devices`

If there's an error kill all rosnodes with
`rosnode kill -a`
But be careful to catch the robot arm because it will collapse.

## collection script:
```source activate py2```
```(py2) datacol@fermat:~/interbotix_ws/src/imitation_learning$ python imitation_learning/run_data_collection.py  experiments/control/widowx/vr_record_applied_actions/conf.py --prefix bww_grasp_pen```

if script cannot be cancelled with ctrl-c use, use ctrl-z

`ps aux | grep -ie semipara | awk '{print $2}' | xargs kill -9`


To count trajectories, for example run:
`(spt) datacol@fermat:~/interbotix_ws/src/private_visual_foresight$ python visual_mpc/utils/file_2_hdf5.py /home/datacol/Documents/data/spt_trainingdata/control/widowx/vr_control/bww_grasp_pen`

Each time edit the `experiments/control/widowx/vr_control/collection_metadata.json` and
change the --prefix for the collection script.


For 03/13
- 500x pick up penguin, screwdriver"
use prefix "bww_grasp_pen" etc.

for 5 times:
- 100x pick up "pen, red_brush, penguin, screwdriver"
use prefix "bww_camconfig1", "bww_camconfig2"....  etc.____
  randomize cam 2-5, leave 1 as it is.


For 03/06
- 500x pick up "pen, red_brush, penguin, screwdriver"
use prefix "bww_grasp_pen" etc.

For 02/25
- 300x pick up "pen, red_brush, penguin, screwdriver"
- 300x "green_brush, green_lego_cube" 
bww_grasp_pen


