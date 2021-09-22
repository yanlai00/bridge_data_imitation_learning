import os
from semiparametrictransfer.utils.general_utils import AttrDict
from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
from semiparametrictransfer.data_sets.multi_dataset_loader import RandomMixingDatasetLoader

#bridge data
sawyer_widowx_rand_mix = AttrDict(
                             dataclass=RandomMixingDatasetLoader,
                             dataconf=AttrDict(
                                 name='sawyer_widowx_rand_mix',
                                 dataset0=[FixLenVideoDataset,
                                 AttrDict(
                                     name='put_in_bowl_sawyer',
                                     random_crop=[48, 64],
                                     image_size_beforecrop=[56, 72],
                                     data_dir=os.environ['DATA'] + '/spt_trainingdata/control/pybullet_sawyer/rand2out6',
                                 )
                                 ],
                                 dataset1=[FixLenVideoDataset,
                                 AttrDict(
                                     name='put_in_bowl_widowx',
                                     random_crop=[48, 64],
                                     image_size_beforecrop=[56, 72],
                                     data_dir=os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/rand2out6obj'
                                 )
                                 ]
                             )
                        )

put_in_bowl_gatorade_widow = AttrDict(
    random_crop=[48, 64],
    image_size_beforecrop=[56, 72],
    data_dir=os.environ['DATA'] + '/spt_trainingdata/control/widowx/sim/pick_only_gatorade/500'
)

put_in_bowl_gatorade_sawyer = AttrDict(
    random_crop=[48, 64],
    image_size_beforecrop=[56, 72],
    data_dir=os.environ['DATA'] + '/spt_trainingdata/control/pybullet_sawyer/pick_only_gatorade/500'
)



validation_data_conf = AttrDict(
    val0=AttrDict(
        dataclass=FixLenVideoDataset,
        dataconf=AttrDict(
            name='put_in_bowl_gatorade_sawyer',
            **put_in_bowl_gatorade_sawyer,
        )
    ),
)


