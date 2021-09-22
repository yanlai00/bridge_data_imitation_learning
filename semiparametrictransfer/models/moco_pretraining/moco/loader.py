# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
import numpy as np
from torchvision.transforms.functional import to_pil_image


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class TwoCropsTransformVideo:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, viewpoint='same', time_difference=None):
        self.base_transform = base_transform

    def __call__(self, x):
        T = x.images.shape[0]
        ncam = x.images.shape[1]
        t = np.random.randint(0, T)
        icam = np.random.randint(0, ncam)
        x = x.images[t, icam]
        x = ((x + 1)/2 * 255.).astype(np.uint8).transpose(1, 2, 0)
        x = to_pil_image(x)
        q = self.base_transform(x)
        k = self.base_transform(x)
        q = q*2 - 1
        k = k*2 - 1
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
