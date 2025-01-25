import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import os

transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    # transforms.ConvertImageDtype(torch.float32),
])

FILTER_ZERO = True
N_TRAIN = 512


def get_train_dataset():
    dataset = dset.MNIST(
        root=os.path.expanduser("~/torch_datasets/mnist"),
        train=True,
        download=True,
        transform=transform_pipeline
    )

    if FILTER_ZERO:
        filtered_indices = [i for i, (_, target) in enumerate(dataset) if target != 0]
        dataset = torch.utils.data.Subset(dataset, filtered_indices)

    if N_TRAIN is not None:
        indices = torch.randperm(len(dataset)).tolist() # shuffle first
        dataset = torch.utils.data.Subset(dataset, indices[:N_TRAIN])

    return dataset

def get_test_dataset():
    dataset = dset.MNIST(
        root=os.path.expanduser("~/torch_datasets/mnist"),
        train=False,
        download=True,
        transform=transform_pipeline
    )

    if FILTER_ZERO:
        filtered_indices = [i for i, (_, target) in enumerate(dataset) if target != 0]
        dataset = torch.utils.data.Subset(dataset, filtered_indices)

    return dataset

def get_val_dataset():
    dataset = dset.MNIST(
        root=os.path.expanduser("~/torch_datasets/mnist"),
        train=False,
        download=True,
        transform=transform_pipeline
    )

    if FILTER_ZERO:
        filtered_indices = [i for i, (_, target) in enumerate(dataset) if target != 0]
        dataset = torch.utils.data.Subset(dataset, filtered_indices)
    return dataset

def get_dataloader(dataset, batch_size=64):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    cap = get_train_dataset()
    print("Number of samples: ", len(cap))
    img, target = cap[4]  # load 4th sample

    print("Image Size: ", img.size())
    print(target)


    