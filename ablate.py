# https://www.youtube.com/watch?v=t4YXpQd5SX4
# https://ar5iv.org/html/2406.07908
# https://zheng-dai.github.io/AblationBasedCounterfactuals/
# https://github.com/zheng-dai/GenEns

import torch
import torch.nn as nn
from architecture import Simple_VAE
from dataset import get_train_dataset, get_dataloader
device = "mps" if torch.backends.mps.is_available() else "cpu"

run = "runs/vae_test1"

net = Simple_VAE().to(device)
net.load_state_dict(torch.load(f"{run}/ckpt/best.pt", weights_only=True))






def get_loss(image, net):
    reconstructed, mean, logvar = net(image)

    loss = net.loss_function(reconstructed, image, mean, logvar)

    return loss

def ablate(image, net,thresh=0.1):
    
    net.eval()
        
    #image.requires_grad = True

    loss = get_loss(image, net)

    loss.backward()

    with torch.no_grad():
        for param in net.parameters():
            if param.grad is not None:
                param.grad[torch.abs(param.grad) > thresh] = 0


trainset = get_train_dataset()

dataloader = get_dataloader(trainset)

for item, _ in dataloader:
    
    before = get_loss(item.to(device), net)

    ablate(item.to(device), net)

    after = get_loss(item.to(device), net)

    print(before, after)

    break
    
    