"""
Sample selection utilities for targeted fine-tuning.
"""
import torch
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from .common import get_loss, Config
from .ablation import reverse_ablate

def select_hard_samples(dataloader, model, threshold: float = Config.DEFAULT_THRESHOLD, easy_instead: bool = False) -> List[torch.Tensor]:
    """
    Select hard or easy samples based on loss difference threshold.
    
    Args:
        dataloader: DataLoader to sample from
        model: VAE model
        threshold: Threshold for loss difference
        easy_instead: Whether to select easy samples instead of hard ones
    
    Returns:
        List of selected sample tensors
    """
    hard_samples = []
    model.eval()
    model.zero_grad()
    
    for image, _ in tqdm(dataloader, leave=False):
        image = image.to(Config.DEVICE)
        model.zero_grad()
        
        before_loss = get_loss(image, model, mse_instead=Config.USE_MSE_INSTEAD)
        reverse_ablate(image, model)
        after_loss = get_loss(image, model, mse_instead=Config.USE_MSE_INSTEAD)
        
        loss_diff = (before_loss - after_loss).abs()
        
        if easy_instead:
            if loss_diff > threshold:
                hard_samples.append(image)
        else:
            if loss_diff < threshold:
                hard_samples.append(image)
    
    model.zero_grad()
    return hard_samples

def select_hard_samples_by_percentile(dataloader, model, target_percentile: float = 0.5, easy_instead: bool = False) -> torch.Tensor:
    """
    Select hard or easy samples based on percentile of loss differences.
    
    Args:
        dataloader: DataLoader to sample from
        model: VAE model
        target_percentile: Target percentile for selection
        easy_instead: Whether to select easy samples instead of hard ones
    
    Returns:
        Tensor of selected samples
    """
    hard_samples = []
    all_samples = []
    loss_diffs = []
    
    model.eval()
    model.zero_grad()
    
    for image, _ in tqdm(dataloader, leave=False):
        image = image.to(Config.DEVICE)
        model.zero_grad()
        
        before_loss = get_loss(image, model, mse_instead=Config.USE_MSE_INSTEAD)
        reverse_ablate(image, model)
        after_loss = get_loss(image, model, mse_instead=Config.USE_MSE_INSTEAD)
        
        loss_diff = (before_loss - after_loss).abs()
        all_samples.append(image)
        loss_diffs.append(loss_diff.item())
    
    loss_diffs = np.array(loss_diffs)
    threshold = np.percentile(loss_diffs, target_percentile * 100)
    
    for i, image in enumerate(all_samples):
        if easy_instead:
            if loss_diffs[i] > threshold:
                hard_samples.append(image)
        else:
            if loss_diffs[i] < threshold:
                hard_samples.append(image)
    
    hard_samples = torch.stack(hard_samples).to(Config.DEVICE)
    del all_samples, loss_diffs
    
    model.zero_grad()
    return hard_samples

def select_random_samples(dataloader, num_samples: int) -> List[torch.Tensor]:
    """
    Select random samples from dataloader.
    
    Args:
        dataloader: DataLoader to sample from
        num_samples: Number of samples to select
    
    Returns:
        List of selected sample tensors
    """
    all_samples = []
    for image, _ in tqdm(dataloader, leave=False):
        all_samples.append(image.to(Config.DEVICE))
    
    if num_samples >= len(all_samples):
        return all_samples
    
    indices = np.random.choice(len(all_samples), num_samples, replace=False)
    return [all_samples[i] for i in indices]
