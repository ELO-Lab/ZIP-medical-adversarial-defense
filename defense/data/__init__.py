import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Subset
import numpy as np
import torchvision
from PIL import Image
from functools import partial


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

def center_crop_arr(pil_image, image_size = 256):
    # Imported from openai/guided-diffusion
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    
    
    if config.data.dataset == 'Attack':
        # only use validation dataset here
        TRANSFORM_IMG = transforms.Compose([
                                transforms.Resize(256),
                             transforms.CenterCrop(256),
                             transforms.ToTensor()])
        dataset = torchvision.datasets.ImageFolder('exp/datasets/BD/', transform=TRANSFORM_IMG)
        test_dataset = dataset
        
    

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X

import torch.nn.functional as F
def wiener_filter(input_tensor, kernel_size=(3, 3)):
    pad_h = kernel_size[0] // 2
    pad_w = kernel_size[1] // 2
    padding = (pad_h, pad_w, pad_h, pad_w)
    
    # Calculate local mean
    local_mean = F.avg_pool2d(input_tensor, kernel_size, stride=1, padding=pad_h)
    
    # Calculate local variance
    local_var = F.avg_pool2d(input_tensor**2, kernel_size, stride=1, padding=pad_h) - local_mean**2
    
    # Estimate noise variance
    noise_var = torch.mean(local_var)
    
    # Apply Wiener filter formula
    filtered_tensor = local_mean + (torch.clamp(local_var - noise_var, min=0) / (local_var + noise_var)) * (input_tensor - local_mean)
    
    return filtered_tensor

def bilateral_filter(input_image, diameter=3, sigma_color=5, sigma_space=3):
    """
    Applies bilateral filter to an input image using PyTorch.

    Args:
    - input_image (torch.Tensor): Input image of shape (C, H, W)
    - diameter (int): Diameter of each pixel neighborhood.
    - sigma_color (float): Filter sigma in the color space.
    - sigma_space (float): Filter sigma in the coordinate space.

    Returns:
    - output_image (torch.Tensor): Bilaterally filtered image of shape (C, H, W)
    """
    if len(input_image.shape) != 3:
        raise ValueError("Input image must have 3 dimensions: (C, H, W)")
    
    channels, height, width = input_image.shape
    output_image = torch.zeros_like(input_image)

    # Create a Gaussian spatial kernel
    d = diameter // 2
    y, x = torch.meshgrid(torch.arange(-d, d + 1), torch.arange(-d, d + 1))
    spatial_kernel = torch.exp(-(x**2 + y**2) / (2 * sigma_space**2))
    spatial_kernel = spatial_kernel / spatial_kernel.sum()
    spatial_kernel = spatial_kernel.to(input_image.device)
    
    # Pad the input image
    padded_image = F.pad(input_image, (d, d, d, d), mode='reflect')

    # Apply bilateral filter
    for i in range(height):
        for j in range(width):
            # Extract local region
            local_region = padded_image[:, i:i + diameter, j:j + diameter]
            
            # Compute color distance kernel
            center_pixel = padded_image[:, i + d, j + d].unsqueeze(1).unsqueeze(2)
            color_distance = torch.exp(-torch.sum((local_region - center_pixel) ** 2, dim=0) / (2 * sigma_color ** 2))
            
            # Combine spatial and color distances
            combined_kernel = spatial_kernel * color_distance
            
            # Normalize the kernel
            combined_kernel = combined_kernel / combined_kernel.sum()
            
            # Apply kernel to local region
            output_image[:, i, j] = (local_region * combined_kernel).sum(dim=(1, 2))

    return output_image

def inverse_data_transform(config, X):
    # X = wiener_filter(X, (3, 3))  
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    # return torch.clamp(wiener_filter(X), 0.0, 1.0)
    # return torch.clamp(bilateral_filter(X), 0.0, 1.0)
    return torch.clamp(X, 0.0, 1.0)
