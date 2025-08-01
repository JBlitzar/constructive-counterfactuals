import os
from fid_metric import compute_fid # import first to set the environment variable for MPS fallback!
import torch
import torch.nn as nn
from architecture import Simple_VAE
from dataset import get_train_dataset, get_dataloader, get_test_dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

os.system(f"caffeinate -is -w {os.getpid()} &")

device = "mps" if torch.backends.mps.is_available() else "cpu"

run = "runs/vae_l5_linear_no0"#"runs/vae_l5_linear_512_no0"

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

def select_hard_samples_by_target(dataloader, net, target_percentile=0.5, easy_instead=False):
    hard_samples = []
    all_samples = []
    loss_diffs = []
    net.eval()
    net.zero_grad()
    for image, _ in tqdm(dataloader, leave=False):
        image = image.to(device)
        net.zero_grad()
        before_loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)
        reverse_ablate(image, net)
        after_loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)
        loss_diff = (before_loss - after_loss).abs()
        all_samples.append(image)
        loss_diffs.append(loss_diff.item())
    
    loss_diffs = np.array(loss_diffs)
    threshold = np.percentile(loss_diffs, target_percentile * 100)
    for i, image in enumerate(all_samples):
        if easy_instead:
            if loss_diffs[i] > threshold:
                hard_samples.append(image)
        else:
            if loss_diffs[i] < threshold:
                hard_samples.append(image)
    hard_samples = torch.stack(hard_samples).to(device)
    del all_samples, loss_diffs

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


percentiles = np.arange(0.01, 1, 0.01).tolist()
print(percentiles)
losses_after, fids_after, losses_zero_after, fids_zero_after = [], [], [], []

for perc in percentiles:
    print(f"\nFine-tuning with target percentile {perc}...")
    hard_samples = select_hard_samples_by_target(dataloader, net, target_percentile=perc, easy_instead=False)
    print(f"Selected {len(hard_samples)} hard samples for fine-tuning at percentile {perc}")

    net.load_state_dict(torch.load(f"{run}/ckpt/best.pt", weights_only=True))
    net.train()
    for epoch in trange(epochs, leave=False):
        for image in hard_samples:
            optimizer.zero_grad()
            loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)
            loss.backward()
            optimizer.step()

    # torch.save(net.state_dict(), f"{run}/ckpt/fine_tuned_{int(perc*100)}.pt")

    print("\nAfter Fine-Tuning:")
    loss_test_after, fid_test_after = evaluate_model(net, get_test_dataset(), "Test")
    loss_test_zero_after, fid_test_zero_after = evaluate_model(net, get_test_dataset(invert_filter=True), "Test (only 0)")

    losses_after.append(loss_test_after)
    fids_after.append(fid_test_after)
    losses_zero_after.append(loss_test_zero_after)
    fids_zero_after.append(fid_test_zero_after)

# Print results
print("\nPercentiles:", percentiles)
print("Loss after:", losses_after)
print("FID after:", fids_after)
print("Loss 0 after:", losses_zero_after)
print("FID 0 after:", fids_zero_after)
# Plotting Loss
plt.figure(figsize=(10, 5))
plt.plot(percentiles, losses_after, marker='o', label='Loss (Test)')
plt.plot(percentiles, losses_zero_after, marker='o', label='Loss (Test only 0)')
plt.xlabel('Target Percentile')
plt.ylabel('Loss')
plt.title('Fine-tuning Loss vs. Target Percentile')
plt.legend()
plt.grid(True)
plt.savefig('results/finetuning_loss_vs_percentile.png')
#plt.show()

# Plotting FID
plt.figure(figsize=(10, 5))
plt.plot(percentiles, fids_after, marker='o', label='FID (Test)')
plt.plot(percentiles, fids_zero_after, marker='o', label='FID (Test only 0)')
plt.xlabel('Target Percentile')
plt.ylabel('FID')
plt.title('Fine-tuning FID vs. Target Percentile')
plt.legend()
plt.grid(True)
plt.savefig('results/finetuning_fid_vs_percentile.png')
#plt.show()

# Random selection comparison
print("\nPerforming random selection comparison...")
def select_random_samples(dataloader, num_samples):
    all_samples = []
    for image, _ in tqdm(dataloader, leave=False):
        all_samples.append(image.to(device))
    
    if num_samples >= len(all_samples):
        return all_samples
    
    indices = np.random.choice(len(all_samples), num_samples, replace=False)
    return [all_samples[i] for i in indices]
losses_random, fids_random, losses_zero_random, fids_zero_random = [], [], [], []

for perc in percentiles:
    print(f"\nFine-tuning with random {perc} proportion...")
    num_samples = int(len(dataloader.dataset) * perc)
    random_samples = select_random_samples(dataloader, num_samples)
    print(f"Selected {len(random_samples)} random samples for fine-tuning")

    net.load_state_dict(torch.load(f"{run}/ckpt/best.pt", weights_only=True))
    net.train()
    for epoch in trange(epochs, leave=False):
        for image in random_samples:
            optimizer.zero_grad()
            loss = get_loss(image, net, mse_instead=USE_MSE_INSTEAD)
            loss.backward()
            optimizer.step()

    print("\nAfter Random Fine-Tuning:")
    loss_test_after, fid_test_after = evaluate_model(net, get_test_dataset(), "Test")
    loss_test_zero_after, fid_test_zero_after = evaluate_model(net, get_test_dataset(invert_filter=True), "Test (only 0)")

    losses_random.append(loss_test_after)
    fids_random.append(fid_test_after)
    losses_zero_random.append(loss_test_zero_after)
    fids_zero_random.append(fid_test_zero_after)

# Plotting Loss with Random comparison
plt.figure(figsize=(10, 5))
plt.plot(percentiles, losses_after, marker='o', label='Loss (Test) - Targeted')
plt.plot(percentiles, losses_zero_after, marker='o', label='Loss (Test only 0) - Targeted')
plt.plot(percentiles, losses_random, marker='s', label='Loss (Test) - Random')
plt.plot(percentiles, losses_zero_random, marker='s', label='Loss (Test only 0) - Random')
plt.xlabel('Proportion of Data')
plt.ylabel('Loss')
plt.title('Fine-tuning Loss: Targeted vs Random Selection')
plt.legend()
plt.grid(True)
plt.savefig('results/comparison_loss_vs_percentile.png')
#plt.show()

# Plotting FID with Random comparison
plt.figure(figsize=(10, 5))
plt.plot(percentiles, fids_after, marker='o', label='FID (Test) - Targeted')
plt.plot(percentiles, fids_zero_after, marker='o', label='FID (Test only 0) - Targeted')
plt.plot(percentiles, fids_random, marker='s', label='FID (Test) - Random')
plt.plot(percentiles, fids_zero_random, marker='s', label='FID (Test only 0) - Random')
plt.xlabel('Proportion of Data')
plt.ylabel('FID')
plt.title('Fine-tuning FID: Targeted vs Random Selection')
plt.legend()
plt.grid(True)
plt.savefig('results/comparison_fid_vs_percentile.png')
#plt.show()




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

