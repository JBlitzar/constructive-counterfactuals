# https://www.youtube.com/watch?v=t4YXpQd5SX4
# https://ar5iv.org/html/2406.07908
# https://zheng-dai.github.io/AblationBasedCounterfactuals/
# https://github.com/zheng-dai/GenEns

# Check out that diff: tensor(6376.6074, device='mps:0', grad_fn=<AddBackward0>) tensor(34779.3555, device='mps:0', grad_fn=<AddBackward0>)

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from architecture import Simple_VAE
from dataset import get_train_dataset, get_dataloader
from utils import (
    Config, load_model, loss_recon_package, visualize_grid,
    ablate, before_after_ablation
)

# Configuration
run = "runs/vae_l5_linear_no0"  # "runs/vae_l5_linear_512_no0"#"runs/vae_512_no0"#"runs/vae_test1"

# Load model
net = load_model(Simple_VAE, run)

def before_after(item, net):
    """Legacy wrapper for backwards compatibility."""
    return before_after_ablation(item, net, ablate, thresh=50, n=10, use_threshold=True, mse_instead=False)

# Main execution
if __name__ == "__main__":
    trainset = get_train_dataset()
    dataloader = get_dataloader(trainset, batch_size=1)

    # Get dummy sample for comparison
    for dummy_item, _ in dataloader:
        dummy = dummy_item.to(Config.DEVICE)
        before_dummy_loss, before_dummy_recon = loss_recon_package(dummy, net)
        break

    # Process first item
    for item, _ in dataloader:
        before_loss, before_recon, after_loss, after_recon = before_after(item.to(Config.DEVICE), net)
        
        item = item.to(Config.DEVICE)
        dummy_loss, dummy_recon = loss_recon_package(dummy, net)

        print(before_loss, after_loss)
        print(before_dummy_loss, dummy_loss)
        
        # Visualize results
        images = [item[0], before_recon[0], after_recon[0], 
                 dummy[0], before_dummy_recon[0], dummy_recon[0]]
        visualize_grid(images, nrow=3, title="Ablation Results")
        break
    
    