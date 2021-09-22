from video_prediction.datasets.sawyer_dataset import SawyerVideoDataset
import tensorflow as tf
import numpy as np
import cv2
import argparse
from visual_mpc.agent.utils.hdf5_saver import HDF5Saver
from semiparametrictransfer.utils.general_utils import AttrDict

batch_size = 1
dataset = SawyerVideoDataset('/mount/harddrive/sawyerdata/annies_data/kinesthetic_demos/tfrecords_combined', mode='train', num_epochs=1)
sess = tf.Session()

inputs, _ = dataset.make_batch(batch_size)

def get_trajectory():
    images = inputs['images']
    images, states, actions, use_action = sess.run([images, inputs['states'], inputs['actions'], inputs['use_action']])
    images = (images * 255).astype(np.uint8)
    return images, states, actions

    # for image in images:
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     cv2.imshow(dataset.input_dir, image)
    #     cv2.waitKey(50)


def save_hdf5(output_folder):
    agentparams = AttrDict(T=15)
    hdf5_saver = HDF5Saver(output_folder, None, agentparams, 1)

    for i in range(1000000):
        print('saving traj', i)
        images, states, actions = get_trajectory()
        env_obs = {'images':images[0][:, None],
                   'state':states[0]}
        action_list = [{'actions': a_t} for a_t in actions[0]]
        hdf5_saver.save_traj(i, {}, obs=env_obs,
                             policy_out=action_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="converts dataset from pkl format to hdf5")
    parser.add_argument('--output_folder', default='', type=str, help='where to save')
    args = parser.parse_args()
    print('saving to ', args.output_folder)
    save_hdf5(args.output_folder)
