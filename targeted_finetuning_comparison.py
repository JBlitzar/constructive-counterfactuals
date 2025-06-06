# filepath: /Users/buckhousefamily/Documents/GitHub/constructive-counterfactuals/targeted_finetuning_comparison.py
import os
from fid_metric import compute_fid  # import first to set the environment variable for MPS fallback!
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from architecture import Simple_VAE
from dataset import get_train_dataset, get_dataloader, get_test_dataset
from utils import (
    Config, load_model, save_model,
    select_hard_samples_by_percentile, select_random_samples,
    fine_tune_model, evaluate_model_metrics
)

# Prevent system sleep during long experiments
os.system(f"caffeinate -is -w {os.getpid()} &")

# Configuration
run = "runs/vae_l5_linear_no0"  # "runs/vae_l5_linear_512_no0"
LEARNING_RATE = 0.00001
epochs = 5

def run_percentile_experiment(percentiles, dataloader, model_class, run_path, test_dataset, test_zero_dataset):
    """Run fine-tuning experiments across different percentiles."""
    losses_after, fids_after, losses_zero_after, fids_zero_after = [], [], [], []
    
    for perc in percentiles:
        print(f"\nFine-tuning with target percentile {perc}...")
        
        # Select samples
        net = load_model(model_class, run_path)
        hard_samples = select_hard_samples_by_percentile(dataloader, net, target_percentile=perc, easy_instead=False)
        print(f"Selected {len(hard_samples)} hard samples for fine-tuning at percentile {perc}")
        
        # Fine-tune
        net = load_model(model_class, run_path)
        optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        fine_tune_model(net, hard_samples, optimizer, epochs)
        
        # Evaluate
        metrics = evaluate_model_metrics(net, test_dataset, test_zero_dataset, "After Fine-Tuning")
        losses_after.append(metrics['loss_test'])
        fids_after.append(metrics['fid_test'])
        losses_zero_after.append(metrics['loss_test_zero'])
        fids_zero_after.append(metrics['fid_test_zero'])
    
    return losses_after, fids_after, losses_zero_after, fids_zero_after

def run_random_comparison(percentiles, dataloader, model_class, run_path, test_dataset, test_zero_dataset):
    """Run random sampling experiments for comparison."""
    losses_random, fids_random, losses_zero_random, fids_zero_random = [], [], [], []
    
    for perc in percentiles:
        print(f"\nFine-tuning with random {perc} proportion...")
        
        # Select samples
        num_samples = int(len(dataloader.dataset) * perc)
        random_samples = select_random_samples(dataloader, num_samples)
        print(f"Selected {len(random_samples)} random samples for fine-tuning")
        
        # Fine-tune
        net = load_model(model_class, run_path)
        optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        fine_tune_model(net, random_samples, optimizer, epochs)
        
        # Evaluate
        metrics = evaluate_model_metrics(net, test_dataset, test_zero_dataset, "After Random Fine-Tuning")
        losses_random.append(metrics['loss_test'])
        fids_random.append(metrics['fid_test'])
        losses_zero_random.append(metrics['loss_test_zero'])
        fids_zero_random.append(metrics['fid_test_zero'])
    
    return losses_random, fids_random, losses_zero_random, fids_zero_random

def plot_results(percentiles, targeted_results, random_results):
    """Plot comparison results."""
    losses_after, fids_after, losses_zero_after, fids_zero_after = targeted_results
    losses_random, fids_random, losses_zero_random, fids_zero_random = random_results
    
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

# Main execution
if __name__ == "__main__":
    # Setup datasets
    trainset = get_train_dataset(invert_filter=True)
    dataloader = get_dataloader(trainset, batch_size=1)
    test_dataset = get_test_dataset()
    test_zero_dataset = get_test_dataset(invert_filter=True)
    
    # Baseline evaluation
    net = load_model(Simple_VAE, run)
    baseline_metrics = evaluate_model_metrics(net, test_dataset, test_zero_dataset, "Before Fine-Tuning")
    
    # Define percentiles for experiments
    percentiles = np.arange(0.1, 1, 0.05).tolist()
    print("Percentiles:", percentiles)
    
    # Run targeted experiments
    print("\n" + "="*50)
    print("RUNNING TARGETED EXPERIMENTS")
    print("="*50)
    targeted_results = run_percentile_experiment(percentiles, dataloader, Simple_VAE, run, test_dataset, test_zero_dataset)
    
    # Run random comparison experiments
    print("\n" + "="*50)
    print("RUNNING RANDOM COMPARISON EXPERIMENTS")
    print("="*50)
    random_results = run_random_comparison(percentiles, dataloader, Simple_VAE, run, test_dataset, test_zero_dataset)
    
    # Print results
    losses_after, fids_after, losses_zero_after, fids_zero_after = targeted_results
    losses_random, fids_random, losses_zero_random, fids_zero_random = random_results
    
    print("\n" + "="*50)
    print("EXPERIMENT RESULTS")
    print("="*50)
    print("Percentiles:", percentiles)
    print("Targeted - Loss after:", losses_after)
    print("Targeted - FID after:", fids_after)
    print("Targeted - Loss 0 after:", losses_zero_after)
    print("Targeted - FID 0 after:", fids_zero_after)
    print("\nRandom - Loss after:", losses_random)
    print("Random - FID after:", fids_random)
    print("Random - Loss 0 after:", losses_zero_random)
    print("Random - FID 0 after:", fids_zero_random)
    
    # Plot results
    plot_results(percentiles, targeted_results, random_results)
    
    # Final full dataset fine-tuning
    print(f"\nFinetuning on full dataset ({len(dataloader.dataset)} samples)...")
    net = load_model(Simple_VAE, run)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    from utils import fine_tune_on_dataloader
    fine_tune_on_dataloader(net, dataloader, optimizer, epochs)
    
    full_metrics = evaluate_model_metrics(net, test_dataset, test_zero_dataset, "After Fine-Tuning on Full Dataset")
    
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"Baseline: Test Loss={baseline_metrics['loss_test']:.4f}, FID={baseline_metrics['fid_test']:.4f}")
    print(f"Full Dataset: Test Loss={full_metrics['loss_test']:.4f}, FID={full_metrics['fid_test']:.4f}")
    print("Plots saved to results/ directory")
