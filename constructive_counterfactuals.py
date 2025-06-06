# filepath: /Users/buckhousefamily/Documents/GitHub/constructive-counterfactuals/constructive_counterfactuals.py
# https://www.youtube.com/watch?v=t4YXpQd5SX4
# https://ar5iv.org/html/2406.07908
# https://zheng-dai.github.io/AblationBasedCounterfactuals/
# https://github.com/zheng-dai/GenEns

# Loss diff with reverse ablation (VAE loss): 170.3783721923828 160.09910583496094
# Loss diff after reverse ablation on non-zero sample (control): 126.66336059570312 127.0302734375

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from architecture import Simple_VAE
from dataset import get_train_dataset, get_dataloader
from utils import (
    Config, load_model, loss_recon_package, visualize_grid,
    reverse_ablate, before_after_ablation
)

# Configuration
run = "runs/vae_l5_linear_no0"

# Load model
net = load_model(Simple_VAE, run)

def before_after(item, net):
    """Legacy wrapper for backwards compatibility."""
    return before_after_ablation(item, net, reverse_ablate, 
                                strength=0.001, mse_instead=Config.USE_MSE_INSTEAD)

# Main execution
if __name__ == "__main__":
    trainset = get_train_dataset(filter_override=True)
    dataloader = get_dataloader(trainset, batch_size=1)

    # Get dummy sample for comparison
    for dummy_item, _ in dataloader:
        dummy = dummy_item.to(Config.DEVICE)
        before_dummy_loss, before_dummy_recon = loss_recon_package(dummy, net, mse_instead=Config.USE_MSE_INSTEAD)
        break

    # Get dummy zero sample
    for dummy_item_zero, label in dataloader:
        if label[0] == 0:
            dummy_zero = dummy_item_zero.to(Config.DEVICE)
            before_dummy_zero_loss, before_dummy_zero_recon = loss_recon_package(dummy_zero, net, mse_instead=Config.USE_MSE_INSTEAD)
            break

    # Process first zero item
    for item, label in dataloader:
        print("loopinginging")
        if label[0] == 0:
            print("found 0")
            before_loss, before_recon, after_loss, after_recon = before_after(item.to(Config.DEVICE), net)

            item = item.to(Config.DEVICE)

            dummy_loss, dummy_recon = loss_recon_package(dummy, net, mse_instead=Config.USE_MSE_INSTEAD)
            dummy_zero_loss, dummy_zero_recon = loss_recon_package(dummy_zero, net, mse_instead=Config.USE_MSE_INSTEAD)

            assert torch.ne(before_recon, after_recon).all()

            print(before_loss, after_loss)
            print(before_dummy_loss, dummy_loss)
            print(before_dummy_zero_loss, dummy_zero_loss)
            
            # Visualize results
            images = [item[0], before_recon[0], after_recon[0], dummy[0], 
                     before_dummy_recon[0], dummy_recon[0], dummy_zero[0],
                     before_dummy_zero_recon[0], dummy_zero_recon[0]]
            visualize_grid(images, nrow=3, title="Constructive Counterfactuals Results")
            break
