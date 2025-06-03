# Neural network architecture for image denoising

import torch
import torch.nn as nn

class DenoisingNetwork(nn.Module):
    """2-layer CNN for noise estimation with configurable channels."""
    
    def __init__(self, in_channels, hidden_channels=64, input_channel_multiplier=1):
        """
        Initialize the denoising network.
        
        Args:
            in_channels (int): Number of input channels
            hidden_channels (int): Number of hidden channels in convolution layers
            input_channel_multiplier (int): Multiplier for input channels
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels * input_channel_multiplier, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1),
        )

    def forward(self, x):
        """Forward pass through the network."""
        return self.layers(x)