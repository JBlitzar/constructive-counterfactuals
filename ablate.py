import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import torch.nn as nn
import torch.optim as optim

def ablation_study(model, dataloader, optimizer, ablation_mask, epochs=10):
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for data, _ in dataloader:
            data = data.view(data.size(0), -1)
            data = data * ablation_mask  # Apply ablation mask
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(dataloader.dataset)}')


# # Ablation study
# ablation_mask = torch.ones(input_dim)
# ablation_mask[::2] = 0  # Example: zero out every other pixel
# ablation_study(model, train_loader, optimizer, ablation_mask, epochs)