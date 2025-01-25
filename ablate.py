# https://www.youtube.com/watch?v=t4YXpQd5SX4
# https://ar5iv.org/html/2406.07908
# https://zheng-dai.github.io/AblationBasedCounterfactuals/
# https://github.com/zheng-dai/GenEns

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
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

def ablate(image, net, thresh=0.01):
    net.eval()
    optimizer = torch.optim.SGD([image], lr=0.1)
    net.zero_grad()
    optimizer.zero_grad()
    
    loss = get_loss(image, net)
    loss.backward()

    for param in net.parameters():
        if param.grad is not None:
            param.grad = -param.grad

    
    optimizer.step()


    

    reconstructed, _, _ = net(image)
    return reconstructed

def visualize_before_after(image, reconstructed):
    # Create a grid of images
    grid = torchvision.utils.make_grid(torch.cat((image, reconstructed), dim=0))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title('Before and After Reconstruction')
    plt.show()

# Example usage
trainset = get_train_dataset()
dataloader = get_dataloader(trainset, batch_size=1)

for image, _ in dataloader:
    image = image.to(device)

    original, _, _ = net(image)
    before = get_loss(image, net)

    ablated = ablate(image, net)
    after = get_loss(image, net)

    print(before.item(), after.item())
    visualize_before_after(original, ablated)
    break