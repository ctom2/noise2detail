# Training functions and loss calculations

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm
from config import LEARNING_RATE, LR_STEP_SIZE, LR_GAMMA, MAX_EPOCHS
from utils import pair_downsample

def calculate_loss(noisy_image, model):
    """
    Compute combined residual and consistency loss for first-stage training.
    
    Args:
        noisy_image (torch.Tensor): Noisy input image
        model (nn.Module): Denoising model
        
    Returns:
        torch.Tensor: Combined loss value
    """
    noisy1, noisy2 = pair_downsample(noisy_image)
    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)
    loss_res = 0.5 * (F.mse_loss(noisy1, pred2) + F.mse_loss(noisy2, pred1))

    denoised = noisy_image - model(noisy_image)
    denoised1, denoised2 = pair_downsample(denoised)
    loss_cons = 0.5 * (F.mse_loss(pred1, denoised1) + F.mse_loss(pred2, denoised2))

    return loss_res + loss_cons

def calculate_loss_2(semi_image, noisy_image, model):
    """
    Compute combined residual and consistency loss for second-stage training.
    
    Args:
        semi_image (torch.Tensor): Semi-denoised image
        noisy_image (torch.Tensor): Original noisy image
        model (nn.Module): Denoising model
        
    Returns:
        torch.Tensor: Combined loss value
    """
    noisy1, noisy2 = pair_downsample(noisy_image)
    semi1, semi2 = pair_downsample(semi_image)
    pred1 = model(semi1)
    pred2 = model(semi2)
    loss_res = 0.5 * (F.mse_loss(noisy1, pred2) + F.mse_loss(noisy2, pred1))

    denoised = model(semi_image)
    denoised1, denoised2 = pair_downsample(denoised)
    loss_cons = 0.5 * (F.mse_loss(pred1, denoised1) + F.mse_loss(pred2, denoised2))

    return loss_res + loss_cons

def train_model(model, noisy_image, epochs=MAX_EPOCHS):
    """
    Training loop for first-stage model with learning rate scheduling.
    
    Args:
        model (nn.Module): Denoising model
        noisy_image (torch.Tensor): Noisy input image
        epochs (int): Number of training epochs
    """
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, LR_STEP_SIZE, LR_GAMMA)

    progress = tqdm(range(epochs), desc="Training Stage 1")
    for _ in progress:
        optimizer.zero_grad()
        loss = calculate_loss(noisy_image, model)
        loss.backward()
        optimizer.step()
        scheduler.step()
        progress.set_postfix(loss=f"{loss.item():.4f}")

def train_model_2(model, semi_image, noisy_image, epochs=MAX_EPOCHS):
    """
    Training loop for second-stage model with learning rate scheduling.
    
    Args:
        model (nn.Module): Denoising model
        semi_image (torch.Tensor): Semi-denoised image
        noisy_image (torch.Tensor): Original noisy image
        epochs (int): Number of training epochs
    """
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, LR_STEP_SIZE, LR_GAMMA)

    progress = tqdm(range(epochs), desc="Training Stage 2")
    for _ in progress:
        optimizer.zero_grad()
        loss = calculate_loss_2(semi_image, noisy_image, model)
        loss.backward()
        optimizer.step()
        scheduler.step()
        progress.set_postfix(loss=f"{loss.item():.4f}")