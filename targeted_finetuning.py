from fid_metric import compute_fid # import first to set the environment variable for MPS fallback!
import torch
import torch.nn as nn
from architecture import Simple_VAE
from dataset import get_train_dataset, get_dataloader, get_test_dataset
import numpy as np
from tqdm import tqdm, trange

device = "mps" if torch.backends.mps.is_available() else "cpu"

run = "runs/vae_l5_linear_no0"

net = Simple_VAE().to(device)
net.load_state_dict(torch.load(f"{run}/ckpt/best.pt", weights_only=True))

USE_MSE_INSTEAD = True
LEARNING_RATE = 0.00001
epochs = 5

trainset = get_train_dataset(invert_filter=True)
dataloader = get_dataloader(trainset, batch_size=1)

optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)


def get_loss(image, net, mse_instead=False):
    reconstructed, mean, logvar = net(image)
    if mse_instead:
        return nn.MSELoss()(reconstructed, image)
    else:
        return net.loss_function(reconstructed, image, mean, logvar)


def reverse_ablate(image, net, strength=0.001):
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
    for image, _ in tqdm(dataloader, leave=False):
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


def evaluate_model(net, dataset, label=""):

    with torch.no_grad():
        net.eval()
        avg_loss = 0
        real_images, generated_images = [], []

        for image, _ in get_dataloader(dataset, batch_size=1):
            image = image.to(device)
            loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)
            avg_loss += loss.item()

            reconstructed, _, _ = net(image)
            real_images.append(image.cpu())
            generated_images.append(reconstructed.cpu())

        avg_loss /= len(dataset)
        real_images = torch.cat(real_images, dim=0).repeat(1, 3, 1, 1)
        generated_images = torch.cat(generated_images, dim=0).repeat(1, 3, 1, 1)
        
        fid_score = compute_fid(real_images, generated_images)

        print(f"Avg Loss on {label} Set: {avg_loss}")
        print(f"FID on {label} Set: {fid_score}")
        return avg_loss, fid_score


print("\nBefore Fine-Tuning:")
loss_test, fid_test = evaluate_model(net, get_test_dataset(), "Test")
loss_test_zero, fid_test_zero = evaluate_model(net, get_test_dataset(invert_filter=True), "Test (only 0)")

easy = False
hard_samples = select_hard_samples(dataloader, net, easy_instead=easy)
print(f"Selected {len(hard_samples)} {'easy' if easy else 'hard'} samples for fine-tuning")

net.load_state_dict(torch.load(f"{run}/ckpt/best.pt", weights_only=True))


net.train()
for epoch in trange(epochs, leave=False):
    for image in hard_samples:
        optimizer.zero_grad()
        loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)
        loss.backward()
        optimizer.step()

torch.save(net.state_dict(), f"{run}/ckpt/fine_tuned.pt")


print("\nAfter Fine-Tuning:")
loss_test_after, fid_test_after = evaluate_model(net, get_test_dataset(), "Test")
loss_test_zero_after, fid_test_zero_after = evaluate_model(net, get_test_dataset(invert_filter=True), "Test (only 0)")

def select_random_samples(dataloader, num_samples):
    all_samples = []
    for image, _ in tqdm(dataloader, leave=False):
        all_samples.append(image.to(device))
    
    if num_samples >= len(all_samples):
        return all_samples
    
    indices = np.random.choice(len(all_samples), num_samples, replace=False)
    return [all_samples[i] for i in indices]


print(f"\nFine-tuning with {len(hard_samples)} randomly selected samples...")
net.load_state_dict(torch.load(f"{run}/ckpt/best.pt", weights_only=True))

random_samples = select_random_samples(dataloader, len(hard_samples))
print(f"Selected {len(random_samples)} random samples for fine-tuning")

net.train()
for epoch in trange(epochs, leave=False):
    for image in random_samples:
        optimizer.zero_grad()
        loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)
        loss.backward()
        optimizer.step()

torch.save(net.state_dict(), f"{run}/ckpt/random_fine_tuned.pt")

# After Random Fine-Tuning Metrics
print("\nAfter Random Fine-Tuning:")
loss_test_random, fid_test_random = evaluate_model(net, get_test_dataset(), "Test")
loss_test_zero_random, fid_test_zero_random = evaluate_model(net, get_test_dataset(invert_filter=True), "Test (only 0)")



print(f"Finetuning on full dataset ({len(dataloader.dataset)} samples)...")
net.load_state_dict(torch.load(f"{run}/ckpt/best.pt", weights_only=True))


net.train()
for epoch in trange(epochs, leave=False):
    for image, _ in dataloader:
        image = image.to(device)
        optimizer.zero_grad()
        loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)
        loss.backward()
        optimizer.step()

print("\nAfter Fine-Tuning on Full Dataset:")
loss_test_full, fid_test_full = evaluate_model(net, get_test_dataset(), "Test")
loss_test_zero_full, fid_test_zero_full = evaluate_model(net, get_test_dataset(invert_filter=True), "Test (only 0)")

