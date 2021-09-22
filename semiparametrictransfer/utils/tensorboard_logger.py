import os
import pdb
import torchvision
from semiparametrictransfer.utils.vis_utils import draw_text_image
from semiparametrictransfer.utils.general_utils import select_indices
import torch
from tensorboardX import SummaryWriter
import numpy as np
import copy
from semiparametrictransfer.utils.general_utils import np_unstack

class Logger:
    def __init__(self, log_dir, n_logged_samples=10, summary_writer=None):
        self._log_dir = log_dir
        self._n_logged_samples = n_logged_samples
        if summary_writer is not None:
            self._summ_writer = summary_writer
        else:
            self._summ_writer = SummaryWriter(log_dir)

    def _loop_batch(self, fn, name, val, *argv, **kwargs):
        """Loops the logging function n times."""
        for log_idx in range(min(self._n_logged_samples, len(val))):
            name_i = os.path.join(name, "_%d" % log_idx)
            fn(name_i, val[log_idx], *argv, **kwargs)

    @staticmethod
    def _check_size(val, size):
        if isinstance(val, torch.Tensor) or isinstance(val, np.ndarray):
            assert len(val.shape) == size, "Size of tensor does not fit required size, {} vs {}".format(len(val.shape),
                                                                                                        size)
        elif isinstance(val, list):
            assert len(val[0].shape) == size - 1, "Size of list element does not fit required size, {} vs {}".format(
                len(val[0].shape), size - 1)
        else:
            raise NotImplementedError("Input type {} not supported for dimensionality check!".format(type(val)))
        if (val[0].shape[1] > 10000) or (val[0].shape[2] > 10000):
            raise ValueError("This might be a bit too much")

    def log_scalar(self, scalar, name, step, phase=None):
        if phase:
            sc_name = '{}_{}'.format(name, phase)
        else:
            sc_name = name
        self._summ_writer.add_scalar(sc_name, scalar, step)

    def log_scalars(self, scalar_dict, group_name, step, phase=None):
        """Will log all scalars in the same plot."""
        if phase:
            sc_name = '{}_{}'.format(group_name, phase)
        else:
            sc_name = group_name
        self._summ_writer.add_scalars('{}_{}'.format(sc_name, phase), scalar_dict, step)

    def log_images(self, image, name, step, phase):
        self._check_size(image, 4)  # [N, C, H, W]
        self._loop_batch(self._summ_writer.add_image, '{}_{}'.format(name, phase), image, step)

    def log_video(self, video_frames, name, step, phase=None, fps=10):
        assert len(video_frames.shape) == 4, "Need [T, C, H, W] input tensor for single video logging!"
        if not isinstance(video_frames, torch.Tensor): video_frames = torch.tensor(video_frames)
        video_frames = video_frames.unsqueeze(0)  # add an extra dimension to get grid of size 1
        if phase:
            sc_name = '{}_{}'.format(name, phase)
        else:
            sc_name = name
        self._summ_writer.add_video(sc_name, video_frames, step, fps=fps)

    def log_image(self, images, name, step, phase):
        self._summ_writer.add_image('{}_{}'.format(name, phase), images, step)

    def log_image_grid(self, images, name, step, phase, nrow=8):
        assert len(images.shape) == 4, "Image grid logging requires input shape [batch, C, H, W]!"
        img_grid = torchvision.utils.make_grid(images, nrow=nrow)
        self.log_images(img_grid, '{}_{}'.format(name, phase), step)

    def log_video_grid(self, video_frames, name, step, phase, fps=3):
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        self._summ_writer.add_video('{}_{}'.format(name, phase), video_frames, step, fps=fps)

    def log_figures(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        assert figure.shape[0] > 0, "Figure logging requires input shape [batch x figures]!"
        self._loop_batch(self._summ_writer.add_figure, '{}_{}'.format(name, phase), figure, step)

    def log_figure(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def dump_scalars(self, log_path=None):
        log_path = os.path.join(self._log_dir, "scalar_data.json") if log_path is None else log_path
        self._summ_writer.export_scalars_to_json(log_path)

    def log_kbest_videos(self, model_output, inputs, losses, step, phase):

        def get_per_example_loss_red():
            loss = torch.mean((model_output.a_pred - inputs.sel_actions)**2, dim=-1)
            no_aux_loss = torch.mean((model_output.a_pred_no_aux - inputs.sel_actions)**2, dim=-1)
            loss_red_perex = (loss - no_aux_loss).cpu().detach().numpy()
            loss_reduction_row = np.stack([draw_text_image(str(r), dtype=np.uint8) for r in loss_red_perex], axis=0)
            T = inputs.best_matches_states.shape[1]
            per_example_loss_row = torch.from_numpy(np.tile(loss_reduction_row[:, None], [1, T, 1, 1, 1]))
            return per_example_loss_row

        goal_img = inputs.images.squeeze()[:, -1]
        vid = assemble_videos_kbestmatches(inputs.current_img, goal_img, inputs.best_matches_images, get_per_example_loss_red())
        self.log_video(vid, 'nearest_neighbors', step, phase, fps=10)

    def flush(self):
        self._summ_writer.flush()

def assemble_videos_kbestmatches(current_img, goal_img, best_matches_images, per_example_loss_red=None, n_batch_examples=10):
    """
    all inputs have to torch tensors!
    :param current_img:
    :param goal_img:
    :param best_matches_images:  [b, t, nbest, row, cols, channel]
    :param per_example_loss_red:
    :param n_batch_examples:
    :return:
    """
    video_rows = []  # list of (b, T, rows, cols, 3)
    T = best_matches_images.shape[1]

    if per_example_loss_red is not None:
        video_rows.append(per_example_loss_red)

    video_rows.append(copy.deepcopy(current_img[:, None].repeat(1, T, 1, 1, 1)))
    video_rows.append(copy.deepcopy(goal_img[:, None].repeat(1, T, 1, 1, 1)))

    for i in range(best_matches_images.shape[2]):
        video_rows.append(best_matches_images[:, :T, i])

    video_rows = [v.cpu().numpy() for v in video_rows]

    videos = np.concatenate(video_rows, axis=2)
    videos = np_unstack(videos, axis=0)
    videos = np.concatenate(videos[:n_batch_examples], axis=2)
    videos = np.transpose(videos, [0, 3, 1, 2])
    return videos


import semiparametrictransfer

class Mujoco_Renderer():
    def __init__(self, im_height, im_width):
        from mujoco_py import load_model_from_path, MjSim

        mujoco_xml = '/'.join(str.split(semiparametrictransfer.__file__, '/')[:-1]) \
                     + '/environments/tabletop/assets/sawyer_xyz/sawyer_multiobject_textured.xml'

        self.sim = MjSim(load_model_from_path(mujoco_xml))
        self.im_height = im_height
        self.im_width = im_width

    def render(self, qpos):
        sim_state = self.sim.get_eef_pose()
        sim_state.qpos[:] = qpos
        sim_state.qvel[:] = np.zeros_like(self.sim.data.qvel)
        self.sim.set_state(sim_state)
        self.sim.forward()

        subgoal_image = self.sim.render(self.im_height, self.im_width, camera_name='cam0')
        # plt.imshow(subgoal_image)
        # plt.savefig('test.png')
        return subgoal_image
