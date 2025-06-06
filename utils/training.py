"""
Training utilities for fine-tuning experiments.
"""
import torch
from typing import List, Dict, Any
from tqdm import trange
from .common import get_loss, save_model, Config

def fine_tune_model(model, samples: List[torch.Tensor], optimizer, epochs: int = Config.DEFAULT_EPOCHS, mse_instead: bool = Config.USE_MSE_INSTEAD):
    """
    Fine-tune model on selected samples.
    
    Args:
        model: VAE model to fine-tune
        samples: List of sample tensors to train on
        optimizer: Optimizer for training
        epochs: Number of training epochs
        mse_instead: Whether to use MSE loss instead of VAE loss
    """
    model.train()
    for epoch in trange(epochs, leave=False):
        for image in samples:
            optimizer.zero_grad()
            loss = get_loss(image, model, mse_instead=mse_instead)
            loss.backward()
            optimizer.step()

def fine_tune_on_dataloader(model, dataloader, optimizer, epochs: int = Config.DEFAULT_EPOCHS, mse_instead: bool = Config.USE_MSE_INSTEAD):
    """
    Fine-tune model on full dataloader.
    
    Args:
        model: VAE model to fine-tune
        dataloader: DataLoader to train on
        optimizer: Optimizer for training
        epochs: Number of training epochs
        mse_instead: Whether to use MSE loss instead of VAE loss
    """
    model.train()
    for epoch in trange(epochs, leave=False):
        for image, _ in dataloader:
            image = image.to(Config.DEVICE)
            optimizer.zero_grad()
            loss = get_loss(image, model, mse_instead=mse_instead)
            loss.backward()
            optimizer.step()

class ExperimentRunner:
    """Class to run fine-tuning experiments with consistent setup."""
    
    def __init__(self, model_class, run_path: str, learning_rate: float = Config.LEARNING_RATE):
        """
        Initialize experiment runner.
        
        Args:
            model_class: Class of the VAE model
            run_path: Path to model checkpoints
            learning_rate: Learning rate for optimizer
        """
        self.model_class = model_class
        self.run_path = run_path
        self.learning_rate = learning_rate
    
    def run_targeted_vs_random_experiment(self, targeted_samples: List[torch.Tensor], 
                                        random_samples: List[torch.Tensor],
                                        test_dataset, test_zero_dataset,
                                        epochs: int = Config.DEFAULT_EPOCHS) -> Dict[str, Any]:
        """
        Run comparison experiment between targeted and random fine-tuning.
        
        Args:
            targeted_samples: Samples selected by targeting strategy
            random_samples: Randomly selected samples
            test_dataset: Test dataset for evaluation
            test_zero_dataset: Zero-only test dataset for evaluation
            epochs: Number of training epochs
        
        Returns:
            Dictionary containing experiment results
        """
        from .common import load_model
        from .evaluation import evaluate_model_metrics
        
        results = {}
        
        # Baseline evaluation
        model = load_model(self.model_class, self.run_path)
        results['baseline'] = evaluate_model_metrics(model, test_dataset, test_zero_dataset, "Baseline")
        
        # Targeted fine-tuning
        print(f"\nTargeted fine-tuning with {len(targeted_samples)} samples...")
        model = load_model(self.model_class, self.run_path)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        fine_tune_model(model, targeted_samples, optimizer, epochs)
        save_model(model, self.run_path, "targeted_fine_tuned.pt")
        results['targeted'] = evaluate_model_metrics(model, test_dataset, test_zero_dataset, "After Targeted Fine-Tuning")
        
        # Random fine-tuning
        print(f"\nRandom fine-tuning with {len(random_samples)} samples...")
        model = load_model(self.model_class, self.run_path)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        fine_tune_model(model, random_samples, optimizer, epochs)
        save_model(model, self.run_path, "random_fine_tuned.pt")
        results['random'] = evaluate_model_metrics(model, test_dataset, test_zero_dataset, "After Random Fine-Tuning")
        
        return results
