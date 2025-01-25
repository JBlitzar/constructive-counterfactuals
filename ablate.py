# https://www.youtube.com/watch?v=t4YXpQd5SX4
# https://ar5iv.org/html/2406.07908
# https://zheng-dai.github.io/AblationBasedCounterfactuals/
# https://github.com/zheng-dai/GenEns

# Check out that diff: tensor(6376.6074, device='mps:0', grad_fn=<AddBackward0>) tensor(34779.3555, device='mps:0', grad_fn=<AddBackward0>)

import torch
import torch.nn as nn
import torchvision
from architecture import Simple_VAE
from dataset import get_train_dataset, get_dataloader
import numpy as np
import matplotlib.pyplot as plt
device = "mps" if torch.backends.mps.is_available() else "cpu"

run = "runs/vae_test1"

net = Simple_VAE().to(device)
net.load_state_dict(torch.load(f"{run}/ckpt/best.pt", weights_only=True))






def get_loss(image, net):
    reconstructed, mean, logvar = net(image)

    loss = net.loss_function(reconstructed, image, mean, logvar)

    return loss


@torch.no_grad()
def loss_recon_package(image, net):
    reconstructed, mean, logvar = net(image)

    loss = net.loss_function(reconstructed, image, mean, logvar)

    return loss.item(), reconstructed

def ablate(image, net,thresh=200,n = 10,use_threshold = False):

    
    
    net.eval()
        
    #image.requires_grad = True

    loss = get_loss(image, net)

    loss.backward()

    with torch.no_grad():
        grads = []
        for param in net.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        
        grads = torch.cat(grads)
        top_n_values, _ = torch.topk(torch.abs(grads), n)
        
        threshold = thresh if use_threshold else top_n_values[-1] 

        for param in net.parameters():
            if param.grad is not None:
                param[torch.abs(param.grad) >= threshold] = 0

def before_after(item, net):
    before_loss, before_recon = loss_recon_package(item, net)

    ablate(item, net)

    after_loss, after_recon = loss_recon_package(item, net)

    return before_loss, before_recon, after_loss, after_recon


trainset = get_train_dataset()

dataloader = get_dataloader(trainset)

for dummy_item, _ in dataloader:
    dummy = dummy_item.to(device)

    before_dummy_loss, before_dummy_recon = loss_recon_package(dummy, net)
    
    break

for item, _ in dataloader:

    before_loss, before_recon, after_loss, after_recon = before_after(item.to(device), net)

    item = item.to(device)

    dummy_loss, dummy_recon = loss_recon_package(dummy, net)


    print(before_loss, after_loss)
    print(before_dummy_loss, dummy_loss)
    grid = torchvision.utils.make_grid([item[0], before_recon[0], after_recon[0], dummy[0], before_dummy_recon[0], dummy_recon[0]], nrow=3).cpu().numpy()
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()


    break
    
    