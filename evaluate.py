# Evaluation and visualization functions

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def evaluate_results(clean, noisy, denoised):
    """
    Calculate and display performance metrics (MSE and PSNR).
    
    Args:
        clean (torch.Tensor): Ground truth image
        noisy (torch.Tensor): Noisy input image
        denoised (torch.Tensor): Denoised output image
        
    Returns:
        tuple: PSNR values for noisy and denoised images
    """
    mse_noisy = F.mse_loss(clean, noisy).item()
    mse_denoised = F.mse_loss(clean, denoised).item()

    psnr_noisy = 10 * np.log10(1 / mse_noisy)
    psnr_denoised = 10 * np.log10(1 / mse_denoised)

    print(f"PSNR Improvement: {psnr_denoised - psnr_noisy:.2f} dB")
    print(f"Final PSNR: {psnr_denoised:.2f} dB")

    return psnr_noisy, psnr_denoised

def plot_results(clean, noisy, denoised, psnr_noisy, psnr_denoised):
    """
    Create visual comparison of ground truth, noisy, and denoised images.
    
    Args:
        clean (torch.Tensor): Ground truth image
        noisy (torch.Tensor): Noisy input image
        denoised (torch.Tensor): Denoised output image
        psnr_noisy (float): PSNR of noisy image
        psnr_denoised (float): PSNR of denoised image
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Ground Truth', 'Noisy Input', 'Denoised Result']
    images = [clean, noisy, denoised]
    psnrs = ['', f'{psnr_noisy:.2f} dB', f'{psnr_denoised:.2f} dB']

    for ax, title, img, psnr in zip(axes, titles, images, psnrs):
        img = img.cpu().squeeze()
        if img.shape[0] == 1:
            ax.imshow(img.squeeze(), cmap='gray')
        else:
            ax.imshow(img.permute(1, 2, 0))
        ax.set_title(title)
        ax.set_xlabel(psnr)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()