# Utility functions for image processing and noise addition

import torch
import torch.nn.functional as F
from config import NOISE_TYPE, NOISE_LEVEL

def add_noise(image, noise_level=NOISE_LEVEL):
    """
    Add synthetic noise to an image.
    
    Args:
        image (torch.Tensor): Input image tensor
        noise_level (float): Noise intensity
        
    Returns:
        torch.Tensor: Noisy image
    """
    if NOISE_TYPE == 'gauss':
        noisy = image + torch.randn_like(image) * (noise_level/255)
        return torch.clamp(noisy, 0, 1)
    
    if NOISE_TYPE == 'poiss':
        return torch.poisson(noise_level * image) / noise_level
    
    raise ValueError(f"Invalid noise type: {NOISE_TYPE}")

def create_downsampling_filters(channels, device):
    """
    Generate fixed downsampling kernels.
    
    Args:
        channels (int): Number of image channels
        device (torch.device): Device to place the filters on
        
    Returns:
        tuple: Two downsampling filter tensors
    """
    filter1 = torch.tensor([[[[0.5, 0.0], [0.0, 0.5]]]], device=device).repeat(channels, 1, 1, 1)
    filter2 = torch.tensor([[[[0.0, 0.5], [0.5, 0.0]]]], device=device).repeat(channels, 1, 1, 1)
    return filter1, filter2

def pair_downsample(image):
    """
    Create downsampled image pair using fixed kernels.
    
    Args:
        image (torch.Tensor): Input image tensor
        
    Returns:
        tuple: Two downsampled images
    """
    channels = image.shape[1]
    filter1, filter2 = create_downsampling_filters(channels, image.device)
    down1 = F.conv2d(image, filter1, stride=2, groups=channels)
    down2 = F.conv2d(image, filter2, stride=2, groups=channels)
    return down1, down2

def pixel_unshuffle(input, factor):
    """
    Rearrange pixels to reduce spatial dimensions while increasing batch size.
    
    Args:
        input (torch.Tensor): Input tensor (n, c, h, w)
        factor (int): Downsampling factor
        
    Returns:
        torch.Tensor: Reshaped tensor (n*factor^2, c, h/factor, w/factor)
    """
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // factor
    out_width = in_width // factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, factor,
        out_width, factor)
    
    batch_size *= factor ** 2
    unshuffle_out = input_view.permute(0, 3, 5, 1, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)

def pixel_shuffle(input, factor):
    """
    Rearrange pixels to increase spatial dimensions while reducing batch size.
    
    Args:
        input (torch.Tensor): Input tensor (n*factor^2, c, h/factor, w/factor)
        factor (int): Upsampling factor
        
    Returns:
        torch.Tensor: Reshaped tensor (n, c, h, w)
    """
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height * factor
    out_width = in_width * factor

    batch_size //= factor ** 2
    batch_size = int(batch_size)
    input_view = input.contiguous().view(
        batch_size, factor, factor, channels, in_height,
        in_width)
    
    unshuffle_out = input_view.permute(0, 3, 4, 1, 5, 2).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)