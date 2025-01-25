# https://www.youtube.com/watch?v=t4YXpQd5SX4
# https://ar5iv.org/html/2406.07908
# https://zheng-dai.github.io/AblationBasedCounterfactuals/
# https://github.com/zheng-dai/GenEns

# Loss diff with reverse ablation: 170.3783721923828 160.09910583496094

# Loss diff after reverse ablation on non-zero sample (control): 126.66336059570312 127.0302734375

import torch
import torch.nn as nn
import torchvision
from architecture import Simple_VAE
from dataset import get_train_dataset, get_dataloader
import numpy as np
import matplotlib.pyplot as plt
device = "mps" if torch.backends.mps.is_available() else "cpu"

run = "runs/vae_l5_linear_512_no0"#"runs/vae_linear_512_no0"#"runs/vae_512_no0"#"runs/vae_test1"

net = Simple_VAE().to(device)
net.load_state_dict(torch.load(f"{run}/ckpt/best.pt", weights_only=True))


USE_MSE_INSTEAD = True



def get_loss(image, net, mse_instead=False):
    reconstructed, mean, logvar = net(image)

    if mse_instead:
        loss = nn.MSELoss()(reconstructed, image)
    else:
        loss = net.loss_function(reconstructed, image, mean, logvar)

    return loss


@torch.no_grad()
def loss_recon_package(image, net, mse_instead=False):
    reconstructed, mean, logvar = net(image)

    if mse_instead:
        loss = nn.MSELoss()(reconstructed, image)
    else:
        loss = net.loss_function(reconstructed, image, mean, logvar)

    return loss.item(), reconstructed

# Name idea: constructive counterfactuals
def reverse_ablate(image, net,strength=0.001):

    
    
    net.eval()

    net.zero_grad()
        
    #image.requires_grad = True

    loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)

    loss.backward()

    with torch.no_grad():
 
        for param in net.parameters():
            if param.grad is not None:
                #param.data = param - param.grad * strength
                param.data = param - torch.sign(param.grad) * strength

def before_after(item, net):
    before_loss, before_recon = loss_recon_package(item, net, mse_instead=USE_MSE_INSTEAD)

    reverse_ablate(item, net)

    after_loss, after_recon = loss_recon_package(item, net, mse_instead=USE_MSE_INSTEAD)

    return before_loss, before_recon, after_loss, after_recon


trainset = get_train_dataset(filter_override=True)

dataloader = get_dataloader(trainset, batch_size=1)

for dummy_item, _ in dataloader:
    dummy = dummy_item.to(device)

    before_dummy_loss, before_dummy_recon = loss_recon_package(dummy, net)
    
    break

for item, label in dataloader:
    print("loopinginging")
    if label[0] == 0:
        print("found 0")
        before_loss, before_recon, after_loss, after_recon = before_after(item.to(device), net)

        item = item.to(device)

        dummy_loss, dummy_recon = loss_recon_package(dummy, net, mse_instead=USE_MSE_INSTEAD)

        assert torch.ne(before_recon, after_recon).all()


        print(before_loss, after_loss)
        print(before_dummy_loss, dummy_loss)
        grid = torchvision.utils.make_grid([item[0], before_recon[0], after_recon[0], dummy[0], before_dummy_recon[0], dummy_recon[0]], nrow=3).cpu().numpy()
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.show()


        break
    
    