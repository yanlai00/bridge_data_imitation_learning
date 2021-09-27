from bridgedata.utils.vis_utils import npy_to_gif, npy_to_mp4
from bridgedata.utils.general_utils import np_unstack
import numpy as np
import os

def make_gifs(loader, outdir=None):
    if outdir is None:
        outdir = os.environ['HOME'] + '/Desktop'
    for i_batch, sample_batched in enumerate(loader):
        images = np.asarray(sample_batched['images'])
        ncam = images.shape[2]
        for cam in range(ncam):
            images_cam = images[:, :, cam]
            images_cam = (np.transpose((images_cam + 1) / 2, [0, 1, 3, 4, 2]) * 255.).astype(np.uint8)  # convert to channel-first

            im_list = []
            for t in range(images_cam.shape[1]):
                im_list.append(np.concatenate(np_unstack(images_cam[:, t], axis=0), axis=1))
            # npy_to_gif(im_list, outdir + '/traj{}_cam_{}'.format(i_batch, cam), fps=10)
            npy_to_mp4(im_list, outdir + '/traj{}_cam_{}'.format(i_batch, cam), fps=10)

        actions = np.asarray(sample_batched['actions'])
        # print('actions', actions)
        print('tlen', sample_batched['tlen'])
        print('camera_ind', sample_batched['camera_ind'])
        # import pdb; pdb.set_trace()

def measure_time(loader):
    import time
    tstart = time.time()
    n_batch = 100
    for i_batch, sample_batched in enumerate(loader):
        print('ibatch', i_batch)
        if i_batch == n_batch:
            break
    print('average loading time', (time.time() - tstart)/n_batch)
