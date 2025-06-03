# Configuration and hyperparameters for the denoising model

# Noise parameters
NOISE_TYPE = 'gauss'  # Options: 'gauss' or 'poiss'
NOISE_LEVEL = 25      # 0-255 for Gaussian, 0-1 for Poisson

# Training parameters
MAX_EPOCHS = 2000
LEARNING_RATE = 0.001
LR_STEP_SIZE = 1000
LR_GAMMA = 0.5
CHAN_EMBED = 48

# Device configuration
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')