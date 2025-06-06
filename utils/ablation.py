"""
Ablation utilities for constructive counterfactuals.
"""
import torch
import torch.nn as nn
from typing import Tuple, List
from .common import get_loss, Config

def ablate(image: torch.Tensor, model, thresh: float = 50, n: int = 10, use_threshold: bool = True):
    """
    Perform ablation on model parameters based on gradients.
    
    Args:
        image: Input image tensor
        model: VAE model
        thresh: Threshold value for ablation
        n: Number of top parameters to ablate
        use_threshold: Whether to use threshold or top-n approach
    """
    model.eval()
    model.zero_grad()
    
    loss = get_loss(image, model)
    loss.backward()
    
    with torch.no_grad():
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        
        grads = torch.cat(grads)
        top_n_values, _ = torch.topk(torch.abs(grads), n)
        
        threshold = thresh if use_threshold else top_n_values[-1]
        
        for param in model.parameters():
            if param.grad is not None:
                param[torch.abs(param.grad) >= threshold] = 0

def reverse_ablate(image: torch.Tensor, model, strength: float = Config.DEFAULT_STRENGTH, mse_instead: bool = Config.USE_MSE_INSTEAD):
    """
    Perform reverse ablation (constructive counterfactual) on model parameters.
    
    Args:
        image: Input image tensor
        model: VAE model
        strength: Strength of the ablation step
        mse_instead: Whether to use MSE loss instead of VAE loss
    """
    model.eval()
    model.zero_grad()
    
    loss = get_loss(image, model, mse_instead=mse_instead)
    loss.backward()
    
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.data = param - torch.sign(param.grad) * strength

def before_after_ablation(item: torch.Tensor, model, ablation_func, **kwargs) -> Tuple[float, torch.Tensor, float, torch.Tensor]:
    """
    Get before and after results for ablation.
    
    Args:
        item: Input image tensor
        model: VAE model
        ablation_func: Function to perform ablation (ablate or reverse_ablate)
        **kwargs: Additional arguments for the ablation function
    
    Returns:
        Tuple of (before_loss, before_recon, after_loss, after_recon)
    """
    from .common import loss_recon_package
    
    # Use the same mse_instead setting for consistency
    mse_instead = kwargs.get('mse_instead', Config.USE_MSE_INSTEAD)
    
    before_loss, before_recon = loss_recon_package(item, model, mse_instead=mse_instead)
    
    ablation_func(item, model, **kwargs)
    
    after_loss, after_recon = loss_recon_package(item, model, mse_instead=mse_instead)
    
    return before_loss, before_recon, after_loss, after_recon
