import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from torcheval.metrics import FrechetInceptionDistance


def compute_fid(images1, images2):
    
    if images1.shape != images2.shape:
        raise ValueError("Both image tensors must have the same shape.")
    
    if images1.ndim != 4 or images2.ndim != 4:
        raise ValueError("Both image tensors must be 4D (batch_size, channels, height, width).")

    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    fid_metric = FrechetInceptionDistance(device=device) # NotImplementedError: The operator 'aten::_linalg_eigvals' is not currently implemented for the MPS device. If you want this op to be added in priority during the prototype phase of this feature, please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.
    # print("initted")
    images1 = images1.clip(0, 1).to(device)
    images2 = images2.clip(0, 1).to(device)

    # print("updating...")

    fid_metric.update(images1, is_real=True)
    fid_metric.update(images2, is_real=False)
    # print("updated. computing...")

    return fid_metric.compute().item()


if __name__ == "__main__":
    real_images = torch.rand(16, 3, 64, 64)
    generated_images = torch.rand(16, 3, 64, 64)

    fid_score = compute_fid(real_images, generated_images)
    print(f"FID Score: {fid_score}")
