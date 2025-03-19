import torch
import torch.nn as nn
import torchvision
from architecture import Simple_VAE
from dataset import get_train_dataset, get_dataloader, get_test_dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
device = "mps" if torch.backends.mps.is_available() else "cpu"

run = "runs/vae_l5_linear_512_no0"

net = Simple_VAE().to(device)
net.load_state_dict(torch.load(f"{run}/ckpt/best.pt", weights_only=True))

USE_MSE_INSTEAD = True
LEARNING_RATE = 0.00001
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


def select_hard_samples(dataloader, net, threshold=0.01, easy_instead=False):
    hard_samples = []
    net.eval()
    net.zero_grad()
    for image, _ in tqdm(dataloader):
        image = image.to(device)
        net.zero_grad()
        before_loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)
        reverse_ablate(image, net)
        after_loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)
        if easy_instead:
            if (before_loss - after_loss).abs() > threshold:
                hard_samples.append(image)
        else:
            if (before_loss - after_loss).abs() < threshold:
                hard_samples.append(image)

    net.zero_grad()
    return hard_samples


with torch.no_grad():
    net.eval()
    avg_loss = 0
    for image, label in get_dataloader(get_test_dataset(), batch_size=1):
        image = image.to(device)
        loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)
        avg_loss += loss.item()
        
    print(f"Avg Loss on Test Set (before training): {avg_loss/len(get_test_dataset())}")

    avg_loss = 0
    for image, label in get_dataloader(get_test_dataset(invert_filter=True), batch_size=1):
        image = image.to(device)
        loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)
        avg_loss += loss.item()
        
    print(f"Avg Loss on Test Set (only 0): {avg_loss/len(get_test_dataset())}")


easy = False
hard_samples = select_hard_samples(dataloader, net, easy_instead=easy)
print(f"Selected {len(hard_samples)} {'easy' if easy else 'hard'} samples for fine-tuning")

net.load_state_dict(torch.load(f"{run}/ckpt/best.pt", weights_only=True)) # haha

net.train()
for epoch in trange(epochs):
    for image in hard_samples:
        optimizer.zero_grad()
        loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)
        loss.backward()
        optimizer.step()
    #print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


torch.save(net.state_dict(), f"{run}/ckpt/fine_tuned.pt")

with torch.no_grad():
    net.eval()
    avg_loss = 0
    for image, label in get_dataloader(get_test_dataset(), batch_size=1):
        image = image.to(device)
        loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)
        avg_loss += loss.item()
        
    print(f"Avg Loss on Test Set: {avg_loss/len(get_test_dataset())}")

    avg_loss = 0
    for image, label in get_dataloader(get_test_dataset(invert_filter=True), batch_size=1):
        image = image.to(device)
        loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)
        avg_loss += loss.item()
        
    print(f"Avg Loss on Test Set (only 0): {avg_loss/len(get_test_dataset())}")


