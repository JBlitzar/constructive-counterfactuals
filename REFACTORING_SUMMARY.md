# Refactoring Summary: Constructive Counterfactuals Codebase

## Overview
The codebase has been successfully refactored to eliminate repetition and create a more modular design. The refactoring focused on the four main files: `ablate.py`, `constructive_counterfactuals.py`, `targeted_finetuning.py`, and `targeted_finetuning_comparison.py`.

## New Modular Structure

### `/utils/` Package
A new utilities package has been created with the following modules:

#### 1. `utils/common.py`
**Purpose**: Common utilities and configurations shared across all modules.

**Key Features**:
- `Config` class: Global configuration settings (device, learning rates, etc.)
- `get_device()`: Automatic device detection (MPS/CPU)
- `load_model()` / `save_model()`: Consistent model loading/saving
- `get_loss()` / `loss_recon_package()`: Unified loss computation
- `visualize_grid()`: Consistent image visualization

#### 2. `utils/ablation.py`
**Purpose**: Ablation utilities for constructive counterfactuals.

**Key Features**:
- `ablate()`: Regular ablation based on gradients
- `reverse_ablate()`: Constructive counterfactual ablation
- `before_after_ablation()`: Generic before/after comparison wrapper

#### 3. `utils/evaluation.py`
**Purpose**: Model evaluation utilities.

**Key Features**:
- `evaluate_model()`: Comprehensive model evaluation (loss + FID)
- `evaluate_model_metrics()`: Batch evaluation for multiple datasets

#### 4. `utils/sampling.py`
**Purpose**: Sample selection utilities for targeted fine-tuning.

**Key Features**:
- `select_hard_samples()`: Threshold-based sample selection
- `select_hard_samples_by_percentile()`: Percentile-based sample selection
- `select_random_samples()`: Random sample selection for comparison

#### 5. `utils/training.py`
**Purpose**: Training utilities for fine-tuning experiments.

**Key Features**:
- `fine_tune_model()`: Fine-tune on selected samples
- `fine_tune_on_dataloader()`: Fine-tune on full dataset
- `ExperimentRunner`: Class for running consistent experiments

## Refactored Files

### 1. `ablate.py` (Before: 107 lines → After: ~50 lines)
- **Eliminated**: Duplicate loss functions, manual device setup, custom visualization
- **Simplified**: Uses modular functions from utils package
- **Maintained**: Original functionality and output format

### 2. `constructive_counterfactuals.py` (Before: 119 lines → After: ~60 lines)
- **Eliminated**: Duplicate loss functions, reverse ablation implementation, visualization code
- **Simplified**: Clean main execution logic
- **Maintained**: All original behavior and assertions

### 3. `targeted_finetuning.py` (Before: 175 lines → After: ~60 lines)
- **Eliminated**: Duplicate evaluation functions, sample selection logic, training loops
- **Added**: Structured experiment runner with summary reporting
- **Simplified**: Clear experiment workflow

### 4. `targeted_finetuning_comparison.py` (Before: 305 lines → After: ~150 lines)
- **Eliminated**: Massive amount of duplicate code
- **Added**: Structured functions for different experiment types
- **Simplified**: Separated experiment logic from plotting logic
- **Improved**: Better organization and error handling

## Benefits of Refactoring

### 1. **Eliminated Repetition**
- **Loss computation**: Previously duplicated 4+ times, now centralized
- **Model loading**: Previously duplicated with slight variations, now consistent
- **Evaluation logic**: Previously copied with minor differences, now unified
- **Sample selection**: Previously repeated with small changes, now modular
- **Training loops**: Previously duplicated, now reusable functions

### 2. **Improved Maintainability**
- **Single source of truth**: Changes to core logic only need to be made once
- **Consistent behavior**: All scripts now use the same underlying functions
- **Easy configuration**: Global config changes apply everywhere
- **Clear separation**: Logic separated from execution

### 3. **Better Code Organization**
- **Logical grouping**: Related functions grouped in appropriate modules
- **Clear dependencies**: Easy to understand what each script depends on
- **Reusable components**: Functions can be easily reused in new experiments
- **Testable units**: Individual functions can be tested independently

### 4. **Enhanced Readability**
- **Reduced complexity**: Main scripts focus on high-level logic
- **Clear intent**: Function names clearly describe what they do
- **Less cognitive load**: Easier to understand what each script does
- **Better documentation**: Centralized documentation in utils modules

## Usage Examples

### Before Refactoring
```python
# Duplicated in every file
device = "mps" if torch.backends.mps.is_available() else "cpu"
net = Simple_VAE().to(device)
net.load_state_dict(torch.load(f"{run}/ckpt/best.pt", weights_only=True))

def get_loss(image, net, mse_instead=False):
    # ... 10 lines of duplicate code
    
def evaluate_model(net, dataset, label=""):
    # ... 25 lines of duplicate code
```

### After Refactoring
```python
# Clean and reusable
from utils import Config, load_model, evaluate_model_metrics

net = load_model(Simple_VAE, run)
metrics = evaluate_model_metrics(net, test_dataset, test_zero_dataset, "Baseline")
```

## Testing
A comprehensive test script (`test_refactoring.py`) has been created to verify:
- ✅ Model loading works correctly
- ✅ Data loading functions properly  
- ✅ Loss computation is accurate
- ✅ Ablation functions work as expected
- ✅ Sample selection operates correctly

## Backward Compatibility
All original scripts maintain their exact same behavior and output format, ensuring no disruption to existing workflows while providing the benefits of cleaner, more maintainable code.

## Future Improvements
The modular structure now makes it easy to:
- Add new ablation techniques
- Implement different evaluation metrics
- Create new sampling strategies
- Add different training approaches
- Extend experiments with minimal code duplication
