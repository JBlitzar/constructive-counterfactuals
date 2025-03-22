import numpy as np
from PIL import Image
from scipy.linalg import sqrtm
from torchvision import transforms
import torch
from torchvision.models import inception_v3
from torcheval.metrics import FrechetInceptionDistance

def preprocess_image(image):
    """Preprocess image to ensure it is 3-channel and 224x224."""
    if len(image.shape) == 2 or image.shape[2] == 1:  # Grayscale or single channel
        image = np.repeat(image, 3, axis=-1)
    image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def calculate_fid(image1, image2):
    """Calculate FID between two images using torcheval."""
    # Preprocess images
    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)

    

    # Initialize FID metric
    fid_metric = FrechetInceptionDistance(feature_dim=2048)
    fid_metric.update(image1, is_real=True)
    fid_metric.update(image2, is_real=False)

    # Compute FID
    fid = fid_metric.compute().item()
    return fid

if __name__ == "__main__":
    # Example usage
    image1 = np.random.rand(224, 224, 3) * 255  # Replace with your actual image
    image2 = np.random.rand(224, 224, 3) * 255  # Replace with your actual image

    fid_value = calculate_fid(image1.astype(np.uint8), image2.astype(np.uint8))
    print(f"FID between the two images: {fid_value}")