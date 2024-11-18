import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps

class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: a solarized image.
        """

        return ImageOps.solarize(img)

class GBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ContrastiveLearningViewGenerator(object):
    def __init__(self, num_patch = 4, scale_min = 0.25, scale_max = 0.25, ratio_min = 1, ratio_max = 1):
    
        self.num_patch = num_patch
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
      
    def __call__(self, x):
    
    
        # normalize = transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(32,scale=(self.scale_min,self.scale_max),ratio=(self.ratio_min,self.ratio_max)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GBlur(p=0.1),
            transforms.RandomApply([Solarization()], p=0.1),
            transforms.ToTensor(),  
        ])
        augmented_x = [aug_transform(x) for i in range(self.num_patch)]
     
        return augmented_x
