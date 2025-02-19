import torch
import torch.nn as nn
import torchvision
from architecture import Simple_VAE
from dataset import get_train_dataset, get_dataloader
import numpy as np
import matplotlib.pyplot as plt

device = "mps" if torch.backends.mps.is_available() else "cpu"

run = "runs/vae_l5_linear_512_no0"

net = Simple_VAE().to(device)
net.load_state_dict(torch.load(f"{run}/ckpt/best.pt", weights_only=True))

USE_MSE_INSTEAD = True
LEARNING_RATE = 1e-4
epochs = 5

# Samples that are *only* 0: Model has never seen these before
# The goal is to fine-tune the model on these samples without using all of them: performing a single FGSM step to identify hard ones, then training on just those.
trainset = get_train_dataset(invert_filter=True)
dataloader = get_dataloader(trainset, batch_size=1)

optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)


def get_loss(image, net, mse_instead=False):
    reconstructed, mean, logvar = net(image)
    if mse_instead:
        return nn.MSELoss()(reconstructed, image)
    else:
        return net.loss_function(reconstructed, image, mean, logvar)


def reverse_ablate(image, net,strength=0.001):
    net.eval()
    net.zero_grad()

    loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)
    loss.backward()

    with torch.no_grad():
        for param in net.parameters():
            if param.grad is not None:
                param.data = param - torch.sign(param.grad) * strength


def select_hard_samples(dataloader, net, threshold=0.01):
    hard_samples = []
    net.eval()
    with torch.no_grad():
        for image, _ in dataloader:
            image = image.to(device)
            before_loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)
            reverse_ablate(image, net)
            after_loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)
            if (before_loss - after_loss).abs() < threshold:
                hard_samples.append(image)
    return hard_samples


hard_samples = select_hard_samples(dataloader, net)
print(f"Selected {len(hard_samples)} hard samples for fine-tuning")


net.train()
for epoch in range(epochs):
    for image in hard_samples:
        optimizer.zero_grad()
        loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


torch.save(net.state_dict(), f"{run}/ckpt/fine_tuned.pt")
