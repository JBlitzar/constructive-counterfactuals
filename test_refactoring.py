"""
Test script to verify the refactored modules work correctly.
"""
import torch
from architecture import Simple_VAE
from dataset import get_train_dataset, get_dataloader
from utils import (
    Config, load_model, get_loss, loss_recon_package,
    ablate, reverse_ablate, evaluate_model, 
    select_hard_samples, select_random_samples
)

def test_basic_functionality():
    """Test basic functionality of refactored modules."""
    print("Testing refactored modules...")
    
    # Test model loading
    print("✓ Testing model loading...")
    run = "runs/vae_l5_linear_no0"
    try:
        net = load_model(Simple_VAE, run)
        print(f"✓ Model loaded successfully on device: {Config.DEVICE}")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False
    
    # Test data loading
    print("✓ Testing data loading...")
    try:
        trainset = get_train_dataset()
        dataloader = get_dataloader(trainset, batch_size=2)
        print(f"✓ Dataset loaded with {len(trainset)} samples")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False
    
    # Test loss computation
    print("✓ Testing loss computation...")
    try:
        for image, _ in dataloader:
            image = image.to(Config.DEVICE)
            loss = get_loss(image, net)
            loss_val, recon = loss_recon_package(image, net)
            print(f"✓ Loss computation successful: {loss_val:.4f}")
            break
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        return False
    
    # Test ablation functions
    print("✓ Testing ablation functions...")
    try:
        # Test reverse ablation
        for image, _ in dataloader:
            image = image.to(Config.DEVICE)
            net_copy = load_model(Simple_VAE, run)  # Fresh copy
            reverse_ablate(image, net_copy)
            print("✓ Reverse ablation successful")
            
            # Test regular ablation
            net_copy2 = load_model(Simple_VAE, run)  # Fresh copy
            ablate(image, net_copy2)
            print("✓ Regular ablation successful")
            break
    except Exception as e:
        print(f"✗ Ablation functions failed: {e}")
        return False
    
    # Test sample selection (small subset)
    print("✓ Testing sample selection...")
    try:
        small_dataloader = get_dataloader(trainset, batch_size=1)
        # Take only first 5 samples for testing
        test_samples = []
        for i, (image, _) in enumerate(small_dataloader):
            if i >= 5:
                break
            test_samples.append((image, _))
        
        # Create a small test dataloader
        from torch.utils.data import DataLoader, TensorDataset
        images = torch.cat([img for img, _ in test_samples])
        labels = torch.cat([torch.tensor([lbl]) for _, lbl in test_samples])
        test_dataset = TensorDataset(images, labels)
        test_dataloader = DataLoader(test_dataset, batch_size=1)
        
        net_fresh = load_model(Simple_VAE, run)
        hard_samples = select_hard_samples(test_dataloader, net_fresh, threshold=0.1)
        print(f"✓ Sample selection successful: {len(hard_samples)} hard samples found")
    except Exception as e:
        print(f"✗ Sample selection failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Refactoring is successful.")
    return True

if __name__ == "__main__":
    test_basic_functionality()
