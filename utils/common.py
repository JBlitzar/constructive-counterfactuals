"""
Common utilities and configurations shared across all modules.
"""
import torch
torch.manual_seed(42) # Set random seed for reproducibility
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union

# Device configuration
def get_device() -> str:
    """Get the appropriate device for computation."""
    return "mps" if torch.backends.mps.is_available() else "cpu"

# Default configuration
class Config:
    """Global configuration settings."""
    DEVICE = get_device()
    USE_MSE_INSTEAD = True
    LEARNING_RATE = 0.00001
    DEFAULT_EPOCHS = 5
    DEFAULT_RUN = "runs/vae_l5_linear_no0"
    DEFAULT_STRENGTH = 0.001
    DEFAULT_THRESHOLD = 0.01

def load_model(model_class, run_path: str, checkpoint_name: str = "best.pt"):
    """
    Load a model from checkpoint.
    
    Args:
        model_class: The model class to instantiate
        run_path: Path to the run directory
        checkpoint_name: Name of the checkpoint file
    
    Returns:
        Loaded model
    """
    device = get_device()
    model = model_class().to(device)
    checkpoint_path = f"{run_path}/ckpt/{checkpoint_name}"
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return model

def save_model(model, run_path: str, checkpoint_name: str):
    """
    Save a model to checkpoint.
    
    Args:
        model: The model to save
        run_path: Path to the run directory
        checkpoint_name: Name of the checkpoint file
    """
    checkpoint_path = f"{run_path}/ckpt/{checkpoint_name}"
    torch.save(model.state_dict(), checkpoint_path)

def get_loss(image: torch.Tensor, model, mse_instead: bool = False) -> torch.Tensor:
    """
    Compute loss for given image and model.
    
    Args:
        image: Input image tensor
        model: VAE model
        mse_instead: Whether to use MSE loss instead of VAE loss
    
    Returns:
        Loss value
    """
    reconstructed, mean, logvar = model(image)
    
    if mse_instead:
        return nn.MSELoss()(reconstructed, image)
    else:
        return model.loss_function(reconstructed, image, mean, logvar)

@torch.no_grad()
def loss_recon_package(image: torch.Tensor, model, mse_instead: bool = False) -> Tuple[float, torch.Tensor]:
    """
    Compute loss and reconstruction for given image.
    
    Args:
        image: Input image tensor
        model: VAE model
        mse_instead: Whether to use MSE loss instead of VAE loss
    
    Returns:
        Tuple of (loss_value, reconstructed_image)
    """
    reconstructed, mean, logvar = model(image)
    
    if mse_instead:
        loss = nn.MSELoss()(reconstructed, image)
    else:
        loss = model.loss_function(reconstructed, image, mean, logvar)
    
    return loss.item(), reconstructed

def visualize_grid(images: List[torch.Tensor], nrow: int = 3, title: str = "", save_path: Optional[str] = None):
    """
    Visualize a grid of images.
    
    Args:
        images: List of image tensors to visualize
        nrow: Number of images per row
        title: Title for the plot
        save_path: Optional path to save the image
    """
    grid = torchvision.utils.make_grid(images, nrow=nrow).cpu().numpy()
    plt.figure(figsize=(12, 8))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
