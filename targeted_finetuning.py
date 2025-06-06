# filepath: /Users/buckhousefamily/Documents/GitHub/constructive-counterfactuals/targeted_finetuning.py
from fid_metric import compute_fid  # import first to set the environment variable for MPS fallback!
import torch
from architecture import Simple_VAE
from dataset import get_train_dataset, get_test_dataset
from utils import (
    Config, load_model, save_model, 
    select_hard_samples, select_random_samples,
    fine_tune_model, evaluate_model_metrics,
    ExperimentRunner
)

# Configuration
run = "runs/vae_l5_linear_no0"
LEARNING_RATE = 0.00001
epochs = 5

# Main execution
if __name__ == "__main__":
    # Setup datasets
    trainset = get_train_dataset(invert_filter=True)
    test_dataset = get_test_dataset()
    test_zero_dataset = get_test_dataset(invert_filter=True)
    
    # Create experiment runner
    runner = ExperimentRunner(Simple_VAE, run, LEARNING_RATE)
    
    # Load model for baseline evaluation
    net = load_model(Simple_VAE, run)
    baseline_metrics = evaluate_model_metrics(net, test_dataset, test_zero_dataset, "Before Fine-Tuning")
    
    # Select samples
    from dataset import get_dataloader
    dataloader = get_dataloader(trainset, batch_size=1)
    
    easy = False
    hard_samples = select_hard_samples(dataloader, net, easy_instead=easy)
    print(f"Selected {len(hard_samples)} {'easy' if easy else 'hard'} samples for fine-tuning")
    
    random_samples = select_random_samples(dataloader, len(hard_samples))
    print(f"Selected {len(random_samples)} random samples for fine-tuning")
    
    # Run targeted vs random experiment
    results = runner.run_targeted_vs_random_experiment(
        hard_samples, random_samples, test_dataset, test_zero_dataset, epochs
    )
    
    # Full dataset fine-tuning
    print(f"\nFinetuning on full dataset ({len(dataloader.dataset)} samples)...")
    net = load_model(Simple_VAE, run)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    from utils import fine_tune_on_dataloader
    fine_tune_on_dataloader(net, dataloader, optimizer, epochs)
    
    full_metrics = evaluate_model_metrics(net, test_dataset, test_zero_dataset, "After Fine-Tuning on Full Dataset")
    
    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Baseline - Test Loss: {baseline_metrics['loss_test']:.4f}, FID: {baseline_metrics['fid_test']:.4f}")
    print(f"Targeted - Test Loss: {results['targeted']['loss_test']:.4f}, FID: {results['targeted']['fid_test']:.4f}")
    print(f"Random   - Test Loss: {results['random']['loss_test']:.4f}, FID: {results['random']['fid_test']:.4f}")
    print(f"Full     - Test Loss: {full_metrics['loss_test']:.4f}, FID: {full_metrics['fid_test']:.4f}")
