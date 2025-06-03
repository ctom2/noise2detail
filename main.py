# Main script for running the denoising pipeline

import torch
import numpy as np
from PIL import Image
import skimage
from model import DenoisingNetwork
from utils import add_noise, pixel_unshuffle, pixel_shuffle
from train import train_model, train_model_2
from evaluate import evaluate_results, plot_results
from config import DEVICE, CHAN_EMBED, MAX_EPOCHS

def main():
    # Load and preprocess image
    clean_img = torch.from_numpy(np.array(skimage.data.astronaut())).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
    clean_img = clean_img / 255
    print(f"Input image shape: {clean_img.shape}")

    # Add noise
    noisy_img = add_noise(clean_img).to(DEVICE)

    # Stage 1: Initial denoising
    model = DenoisingNetwork(clean_img.shape[1], CHAN_EMBED).to(DEVICE)
    train_model(model, noisy_img)

    with torch.no_grad():
        denoised_img_1 = torch.clamp(noisy_img - model(noisy_img), 0, 1)

    # Evaluate and visualize stage 1
    psnr_n, psnr_d = evaluate_results(clean_img, noisy_img, denoised_img_1)
    plot_results(clean_img, noisy_img, denoised_img_1, psnr_n, psnr_d)

    # Stage 2: Ensemble with pixel shuffling
    with torch.no_grad():
        input_2 = pixel_unshuffle(noisy_img, 2)
        pseudo1 = torch.clamp(input_2 - model(input_2), 0, 1)
        pseudo1 = pixel_shuffle(pseudo1, 2)
        pseudo1 = torch.clamp(pseudo1 - model(pseudo1), 0, 1)

        input_4 = pixel_unshuffle(noisy_img, 4)
        pseudo2 = torch.clamp(input_4 - model(input_4), 0, 1)
        pseudo2 = pixel_shuffle(pseudo2, 4)
        pseudo2 = torch.clamp(pseudo2 - model(pseudo2), 0, 1)

        denoised_img_2 = (denoised_img_1 + pseudo1 + pseudo2) / 3

    # Evaluate and visualize stage 2
    psnr_n, psnr_d = evaluate_results(clean_img, noisy_img, denoised_img_2)
    plot_results(clean_img, noisy_img, denoised_img_2, psnr_n, psnr_d)

    # Stage 3: Refine with second model
    model2 = DenoisingNetwork(clean_img.shape[1], CHAN_EMBED).to(DEVICE)
    train_model_2(model2, denoised_img_2, noisy_img)

    with torch.no_grad():
        denoised_img_3 = torch.clamp(model2(denoised_img_2), 0, 1)

    # Evaluate and visualize stage 3
    psnr_n, psnr_d = evaluate_results(clean_img, noisy_img, denoised_img_3)
    plot_results(clean_img, noisy_img, denoised_img_3, psnr_n, psnr_d)

    # Save results
    Image.fromarray((noisy_img.cpu()[0].permute(1,2,0).numpy() * 255).astype(np.uint8)).save('input.png')
    Image.fromarray((denoised_img_3.cpu()[0].permute(1,2,0).numpy() * 255).astype(np.uint8)).save('output.png')

if __name__ == "__main__":
    main()