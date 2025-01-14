import torchvision.datasets as dset
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader
import glob
from PIL import Image
import os



transforms = v2.Compose([
    #v2.PILToTensor(),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomHorizontalFlip(p=0.5),
    #v2.Resize((w,h))
    #v2.CenterCrop(size=(w,h))
    

])

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=False):
        self.root = root_dir
        self.transform = transform
        self.file_list = glob.glob(os.path.join(root_dir, '*'))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image
class TinyImageNetDataset(Dataset):
    def __init__(self, root, transform=None, split="train"):
        root = os.path.expanduser(root)
        self.root_dir = root
        self.transform = transform
        
        
        self.subfolders = glob.glob(os.path.join(os.path.join(root, split), '*'))
        self.all_files = glob.glob(os.path.join(os.path.join(root, split), '*/images/*.JPEG'))

        with open(os.path.join(root, "wnids.txt"), "r") as f:
            wnids = f.read().split("\n")
            self.wnids = {}
            with open(os.path.join(root, "clsloc.txt"), "r") as clsloc:
                clsloc_dict = dict(
                    [
                        [
                            a.split(" ")[0],
                            a.split(" ")[1:3]
                        ] 
                        for a in clsloc.read().split('\n')
                    ]
                    )
                for idx, id in enumerate(wnids):
                    try:
                        self.wnids[id] = [torch.tensor(int(clsloc_dict[id][0])).type(torch.IntTensor), clsloc_dict[id][1]]
                    except (KeyError, IndexError):
                        self.wnids[id] = [torch.tensor(0).type(torch.IntTensor), "NOT_FOUND"]
        
        self.total = len(self.all_files)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        img_path = self.all_files[idx]
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        label = img_path.split("/")[-1].split("_")[0]

        return image, self.wnids[label][0], self.wnids[label][1]

#TODO: fix up dataloader
def get_train_dataset():
    dataset = dset.MNIST(
        root_dir=os.path.expanduser("~/torch_datasets/mnist"),
        train=True,
        transform=transforms
    )
    return dataset

def get_test_dataset():
    dataset = dset.MNIST(
        root_dir=os.path.expanduser("~/torch_datasets/mnist"),
        train=False,
        transform=transforms
    )
    return dataset

def get_val_dataset():

    dataset = dset.MNIST(
        root_dir=os.path.expanduser("~/torch_datasets/mnist"),
        transform=transforms,
        train=False
    )
    return dataset

def get_dataloader(dataset, batch_size=64):

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    cap = get_train_dataset()
    print('Number of samples: ', len(cap))
    img, target = cap[4] # load 4th sample

    print("Image Size: ", img.size())
    print(target)