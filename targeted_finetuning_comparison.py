import os
from fid_metric import (
    compute_fid,
)  # import first to set the environment variable for MPS fallback!
import torch
import torch.nn as nn
from architecture import Simple_VAE
from dataset import get_train_dataset, get_dataloader, get_test_dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd

os.system(f"caffeinate -is -w {os.getpid()} &")


def print(s, *args, **kwargs):
    if args or kwargs:
        s = " ".join([str(s)] + list([str(i) for i in args]))

    tqdm.write(str(s))


device = "mps" if torch.backends.mps.is_available() else "cpu"

run = "runs/vae_l5_linear_no0"  # "runs/vae_l5_linear_512_no0"

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


def select_hard_samples_by_target(
    dataloader, net, target_percentile=0.5, easy_instead=False
):
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
        all_samples.append(image.detach())  # Add .detach() to break gradients
        loss_diffs.append(loss_diff.item())

    loss_diffs = np.array(loss_diffs)
    threshold = np.percentile(loss_diffs, target_percentile * 100)

    # Create hard_samples list without stacking all samples first
    selected_samples = []
    for i, image in enumerate(all_samples):
        if easy_instead:
            if loss_diffs[i] > threshold:
                selected_samples.append(image.detach())
        else:
            if loss_diffs[i] < threshold:
                selected_samples.append(image.detach())

    # Only stack the selected samples
    if selected_samples:
        hard_samples = torch.stack(selected_samples)
    else:
        hard_samples = torch.empty(0, *all_samples[0].shape[1:]).to(device)

    # Explicitly delete large objects and clear cache
    del all_samples, loss_diffs, selected_samples
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

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
            real_images.append(image.detach().cpu())  # Add .detach()
            generated_images.append(reconstructed.detach().cpu())  # Add .detach()

        avg_loss /= len(dataset)
        real_images = torch.cat(real_images, dim=0).repeat(1, 3, 1, 1)
        generated_images = torch.cat(generated_images, dim=0).repeat(1, 3, 1, 1)

        fid_score = compute_fid(real_images, generated_images)

        # Clean up memory
        del real_images, generated_images
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        print(f"Avg Loss on {label} Set: {avg_loss}")
        print(f"FID on {label} Set: {fid_score}")
        return avg_loss, fid_score


print("\nBefore Fine-Tuning:")
loss_test, fid_test = evaluate_model(net, get_test_dataset(), "Test")
loss_test_zero, fid_test_zero = evaluate_model(
    net, get_test_dataset(invert_filter=True), "Test (only 0)"
)


percentiles = np.arange(0.01, 1, 0.01).tolist()
print(percentiles)
losses_after, fids_after, losses_zero_after, fids_zero_after = [], [], [], []

for perc in tqdm(percentiles):
    print(f"\nFine-tuning with target percentile {perc}...")
    hard_samples = select_hard_samples_by_target(
        dataloader, net, target_percentile=perc, easy_instead=False
    )
    print(
        f"Selected {len(hard_samples)} hard samples for fine-tuning at percentile {perc}"
    )

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
    loss_test_zero_after, fid_test_zero_after = evaluate_model(
        net, get_test_dataset(invert_filter=True), "Test (only 0)"
    )

    losses_after.append(loss_test_after)
    fids_after.append(fid_test_after)
    losses_zero_after.append(loss_test_zero_after)
    fids_zero_after.append(fid_test_zero_after)

    # Clean up after each percentile
    del hard_samples
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

# Print results
print("\nPercentiles:", percentiles)
print("Loss after:", losses_after)
print("FID after:", fids_after)
print("Loss 0 after:", losses_zero_after)
print("FID 0 after:", fids_zero_after)
# Plotting Loss
plt.figure(figsize=(10, 5))
plt.plot(percentiles, losses_after, marker="o", label="Loss (Test)")
plt.plot(percentiles, losses_zero_after, marker="o", label="Loss (Test only 0)")
plt.xlabel("Target Percentile")
plt.ylabel("Loss")
plt.title("Fine-tuning Loss vs. Target Percentile")
plt.legend()
plt.grid(True)
plt.savefig("results/finetuning_loss_vs_percentile.png")
# plt.show()

# Plotting FID
plt.figure(figsize=(10, 5))
plt.plot(percentiles, fids_after, marker="o", label="FID (Test)")
plt.plot(percentiles, fids_zero_after, marker="o", label="FID (Test only 0)")
plt.xlabel("Target Percentile")
plt.ylabel("FID")
plt.title("Fine-tuning FID vs. Target Percentile")
plt.legend()
plt.grid(True)
plt.savefig("results/finetuning_fid_vs_percentile.png")
# plt.show()

# Random selection comparison
print("\nPerforming random selection comparison...")


def select_random_samples(dataloader, num_samples):
    all_samples = []
    for image, _ in tqdm(dataloader, leave=False):
        all_samples.append(image.detach().to(device))  # Add .detach()

    if num_samples >= len(all_samples):
        selected = all_samples
    else:
        indices = np.random.choice(len(all_samples), num_samples, replace=False)
        selected = [all_samples[i] for i in indices]

    # Clean up the full list
    del all_samples
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return selected


losses_random, fids_random, losses_zero_random, fids_zero_random = [], [], [], []

for perc in tqdm(percentiles):
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
    loss_test_zero_after, fid_test_zero_after = evaluate_model(
        net, get_test_dataset(invert_filter=True), "Test (only 0)"
    )

    losses_random.append(loss_test_after)
    fids_random.append(fid_test_after)
    losses_zero_random.append(loss_test_zero_after)
    fids_zero_random.append(fid_test_zero_after)

    # Clean up after each percentile
    del random_samples
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

print("\nRandom Selection Results:")
print("Loss after (random):", losses_random)
print("FID after (random):", fids_random)
print("Loss 0 after (random):", losses_zero_random)
print("FID 0 after (random):", fids_zero_random)

# Plotting Loss with Random comparison
plt.figure(figsize=(10, 5))
plt.plot(percentiles, losses_after, marker="o", label="Loss (Test) - Targeted")
plt.plot(
    percentiles, losses_zero_after, marker="o", label="Loss (Test only 0) - Targeted"
)
plt.plot(percentiles, losses_random, marker="s", label="Loss (Test) - Random")
plt.plot(
    percentiles, losses_zero_random, marker="s", label="Loss (Test only 0) - Random"
)
plt.xlabel("Proportion of Data")
plt.ylabel("Loss")
plt.title("Fine-tuning Loss: Targeted vs Random Selection")
plt.legend()
plt.grid(True)
plt.savefig("results/comparison_loss_vs_percentile.png")
# plt.show()

# Plotting FID with Random comparison
plt.figure(figsize=(10, 5))
plt.plot(percentiles, fids_after, marker="o", label="FID (Test) - Targeted")
plt.plot(percentiles, fids_zero_after, marker="o", label="FID (Test only 0) - Targeted")
plt.plot(percentiles, fids_random, marker="s", label="FID (Test) - Random")
plt.plot(percentiles, fids_zero_random, marker="s", label="FID (Test only 0) - Random")
plt.xlabel("Proportion of Data")
plt.ylabel("FID")
plt.title("Fine-tuning FID: Targeted vs Random Selection")
plt.legend()
plt.grid(True)
plt.savefig("results/comparison_fid_vs_percentile.png")
# plt.show()

# Clean up matplotlib figures to free memory
plt.close("all")

# Get the number of samples from the last targeted run for comparison
last_targeted_samples_count = int(len(dataloader.dataset) * percentiles[-1])

print(f"\nFine-tuning with {last_targeted_samples_count} randomly selected samples...")
net.load_state_dict(torch.load(f"{run}/ckpt/best.pt", weights_only=True))

random_samples = select_random_samples(dataloader, last_targeted_samples_count)
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
loss_test_zero_random, fid_test_zero_random = evaluate_model(
    net, get_test_dataset(invert_filter=True), "Test (only 0)"
)

# Clean up
del random_samples
torch.cuda.empty_cache() if torch.cuda.is_available() else None
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.empty_cache()

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
loss_test_zero_full, fid_test_zero_full = evaluate_model(
    net, get_test_dataset(invert_filter=True), "Test (only 0)"
)

# Save results to CSV

results_df = pd.DataFrame(
    {
        "percentile": percentiles,
        "loss_after_targeted": losses_after,
        "fid_after_targeted": fids_after,
        "loss_zero_after_targeted": losses_zero_after,
        "fid_zero_after_targeted": fids_zero_after,
        "loss_after_random": losses_random,
        "fid_after_random": fids_random,
        "loss_zero_after_random": losses_zero_random,
        "fid_zero_after_random": fids_zero_random,
    }
)

# Add baseline and final results
baseline_results = pd.DataFrame(
    {
        "method": ["baseline", "random_final", "full_dataset"],
        "loss_test": [loss_test, loss_test_random, loss_test_full],
        "fid_test": [fid_test, fid_test_random, fid_test_full],
        "loss_test_zero": [loss_test_zero, loss_test_zero_random, loss_test_zero_full],
        "fid_test_zero": [fid_test_zero, fid_test_zero_random, fid_test_zero_full],
    }
)

# Save to CSV files
os.makedirs("results", exist_ok=True)
results_df.to_csv("results/finetuning_comparison_results.csv", index=False)
baseline_results.to_csv("results/baseline_and_final_results.csv", index=False)

print("\nResults saved to CSV files in results/ directory")
