"""
Evaluation utilities for model assessment.
"""
import torch
import numpy as np
from typing import Tuple, List
from tqdm import tqdm
from dataset import get_dataloader
from fid_metric import compute_fid
from .common import get_loss, Config

def evaluate_model(model, dataset, label: str = "", mse_instead: bool = Config.USE_MSE_INSTEAD) -> Tuple[float, float]:
    """
    Evaluate model performance on a dataset.
    
    Args:
        model: VAE model to evaluate
        dataset: Dataset to evaluate on
        label: Label for printing results
        mse_instead: Whether to use MSE loss instead of VAE loss
    
    Returns:
        Tuple of (average_loss, fid_score)
    """
    with torch.no_grad():
        model.eval()
        avg_loss = 0
        real_images, generated_images = [], []
        
        for image, _ in get_dataloader(dataset, batch_size=1):
            image = image.to(Config.DEVICE)
            loss = get_loss(image, model, mse_instead=mse_instead)
            avg_loss += loss.item()
            
            reconstructed, _, _ = model(image)
            real_images.append(image.cpu())
            generated_images.append(reconstructed.cpu())
        
        avg_loss /= len(dataset)
        real_images = torch.cat(real_images, dim=0).repeat(1, 3, 1, 1)
        generated_images = torch.cat(generated_images, dim=0).repeat(1, 3, 1, 1)
        
        fid_score = compute_fid(real_images, generated_images)
        
        if label:
            print(f"Avg Loss on {label} Set: {avg_loss}")
            print(f"FID on {label} Set: {fid_score}")
        
        return avg_loss, fid_score

def evaluate_model_metrics(model, test_dataset, test_zero_dataset, label_prefix: str = "") -> dict:
    """
    Evaluate model on both test datasets and return comprehensive metrics.
    
    Args:
        model: VAE model to evaluate
        test_dataset: Full test dataset
        test_zero_dataset: Zero-only test dataset
        label_prefix: Prefix for labels (e.g., "Before", "After")
    
    Returns:
        Dictionary containing all metrics
    """
    print(f"\n{label_prefix} Evaluation:")
    loss_test, fid_test = evaluate_model(model, test_dataset, "Test")
    loss_test_zero, fid_test_zero = evaluate_model(model, test_zero_dataset, "Test (only 0)")
    
    return {
        'loss_test': loss_test,
        'fid_test': fid_test,
        'loss_test_zero': loss_test_zero,
        'fid_test_zero': fid_test_zero
    }
