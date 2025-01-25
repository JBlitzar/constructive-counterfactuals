import os

os.system(f"caffeinate -is -w {os.getpid()} &")

from architecture import Simple_VAE
from dataset import get_train_dataset, get_test_dataset, get_dataloader
import torch
from tqdm import tqdm, trange
from logger import init_logger
import torchvision
from trainingmanager import TrainingManager


# TODO: setup experiment dir
EXPERIMENT_DIRECTORY = "runs/vae_test1"


device = "mps" if torch.backends.mps.is_available() else "cpu"

dataloader = get_dataloader(get_train_dataset())

testloader = get_dataloader(get_test_dataset())


net = Simple_VAE()
net.to(device)

# TODO: configure
trainer = TrainingManager(
    net=net,
    dir=EXPERIMENT_DIRECTORY,
    dataloader=dataloader,
    device=device,
    trainstep_checkin_interval=100,
    epochs=100,
)


for batch, _ in dataloader:
    init_logger(
        net,
        (batch.to(device)),
        dir=os.path.join(EXPERIMENT_DIRECTORY, "tensorboard"),
    )
    break

trainer.train()
