import matplotlib.pyplot as plt
import torch
import numpy as np
import moviepy.editor as mpy
import os
from PIL import Image

def fig2img(fig):
    """Converts a given figure handle to a 3-channel numpy image array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    w, h, d = buf.shape
    return np.array(Image.frombytes("RGBA", (w, h), buf.tostring()), dtype=np.float32)[:, :, :3] / 255.


def fig2img_(fig):
    """Converts a given figure handle to a 3-channel numpy image array."""
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)

    return buf


def plot_graph(array, h=400, w=400, dpi=10, linewidth=3.0):
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    plt.xlim(0, array.shape[0] - 1)
    plt.xticks(fontsize=100)
    plt.yticks(fontsize=100)
    plt.plot(array)
    plt.grid()
    plt.tight_layout()
    fig_img = fig2img(fig)
    plt.close(fig)
    return fig_img

def plot_bar(array, h=400, w=400, dpi=10, linewidth=3.0):
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    plt.xlim(0, array.shape[0] - 1)
    plt.xticks(fontsize=100)
    plt.yticks(fontsize=100)
    plt.bar(np.arange(array.shape[0]), array)
    plt.grid()
    plt.tight_layout()
    fig_img = fig2img(fig)
    plt.close(fig)
    return fig_img

def npy_to_gif(im_list, filename, fps=4):
    save_dir = '/'.join(str.split(filename, '/')[:-1])
    if not os.path.exists(save_dir):
        print('creating directory: ', save_dir)
        os.makedirs(save_dir)
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')

def npy_to_mp4(im_list, filename, fps=4):
    save_dir = '/'.join(str.split(filename, '/')[:-1])

    if not os.path.exists(save_dir):
        print('creating directory: ', save_dir)
        os.mkdir(save_dir)

    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_videofile(filename + '.mp4')


from PIL import Image, ImageDraw

def draw_text_image(text, background_color=(255,255,255), image_size=(30, 64), dtype=np.float32):

    text_image = Image.new('RGB', image_size[::-1], background_color)
    draw = ImageDraw.Draw(text_image)
    if text:
        draw.text((4, 0), text, fill=(0, 0, 0))
    if dtype == np.float32:
        return np.array(text_image).astype(np.float32)/255.
    else:
        return np.array(text_image)


def draw_text_onimage(text, image, color=(255, 0, 0)):
    if image.dtype == np.float32:
        image = (image*255.).astype(np.uint8)
    assert image.dtype == np.uint8
    from PIL import Image, ImageDraw
    text_image = Image.fromarray(image)
    draw = ImageDraw.Draw(text_image)
    draw.text((4, 0), text, fill=color)
    return np.array(text_image).astype(np.float32)/255.

import cv2

def visualize_barplot_array(input_arr, img_size=(64, 64)):
    plt.switch_backend('agg')
    imgs = []
    for b in range(input_arr.shape[0]):
        img = plot_bar(input_arr[b])
        img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)
        imgs.append((img*255.).astype(np.uint8))
        # cv2.imwrite('/nfs/kun1/users/febert/data/vmpc_exp/test_cv2.png', imgs[-1])
    return imgs

if __name__ == '__main__':
    sigmodis = np.random.random_integers(0, 1, [10, 10])
    visualize_barplot_array(sigmodis)

    # plt.switch_backend('agg')
    # img = plot_graph(np.arange(10))
    # print(np.max(img))
    # print(np.min(img))
    # import pdb; pdb.set_trace()