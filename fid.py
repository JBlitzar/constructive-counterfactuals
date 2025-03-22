import numpy as np
from PIL import Image
from scipy.linalg import sqrtm
from torchvision import transforms
import torch
from torchvision.models import inception_v3

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
    """Calculate FID between two images."""
    # Load InceptionV3 model
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # Remove final classification layer
    model.eval()

    # Preprocess images
    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)

    # Extract features
    with torch.no_grad():
        features1 = model(image1).numpy().squeeze()
        features2 = model(image2).numpy().squeeze()

    # Calculate mean and covariance
    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

    # Calculate FID
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid