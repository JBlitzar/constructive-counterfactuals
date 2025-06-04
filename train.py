import os

os.system(f"caffeinate -is -w {os.getpid()} &")

from architecture import Simple_VAE
from dataset import get_train_dataset, get_test_dataset, get_dataloader
import torch
from tqdm import tqdm, trange
from logger import init_logger
import torchvision
from trainingmanager import TrainingManager



EXPERIMENT_DIRECTORY = "runs/vae_l5_linear_no0"


device = "mps" if torch.backends.mps.is_available() else "cpu"

dataloader = get_dataloader(get_train_dataset())

testloader = get_dataloader(get_test_dataset())


net = Simple_VAE()
net.to(device)


trainer = TrainingManager(
    net=net,
    dir=EXPERIMENT_DIRECTORY,
    dataloader=dataloader,
    val_dataloader=testloader,
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
