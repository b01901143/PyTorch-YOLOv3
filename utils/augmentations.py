import torch
import torch.nn.functional as F
import numpy as np


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

def gaussian_noise(images):
    images = images + torch.normal(torch.zeros(images.shape), torch.ones(images.shape))
    images[images<0] = 0
    images[images>127] = 127
    return images
