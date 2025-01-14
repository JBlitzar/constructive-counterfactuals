import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import os

transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    # transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_train_dataset():
    dataset = dset.MNIST(
        root=os.path.expanduser("~/torch_datasets/mnist"),
        train=True,
        download=True,
        transform=transform_pipeline
    )
    return dataset

def get_test_dataset():
    dataset = dset.MNIST(
        root=os.path.expanduser("~/torch_datasets/mnist"),
        train=False,
        download=True,
        transform=transform_pipeline
    )
    return dataset

def get_val_dataset():
    dataset = dset.MNIST(
        root=os.path.expanduser("~/torch_datasets/mnist"),
        train=False,
        download=True,
        transform=transform_pipeline
    )
    return dataset

def get_dataloader(dataset, batch_size=64):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    cap = get_train_dataset()
    print("Number of samples: ", len(cap))
    img, target = cap[4]  # load 4th sample

    print("Image Size: ", img.size())
    print(target)
