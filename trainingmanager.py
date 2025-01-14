import os
import torch
from logger import log_data, init_logger, log_img
import torch.nn as nn
from tqdm import tqdm, trange


device = "mps" if torch.backends.mps.is_available() else "cpu"


from collections import defaultdict

class ValueTracker:
    def __init__(self):
        self.data = {}
    
    def add(self, label, value):
        if label not in self.data:
            self.data[label] = []
        self.data[label].append(value)
    
    def average(self, label):
        values = self.data[label]
        if values:
            return sum(values) / len(values)
        else:
            return 0.0
    
    def reset(self, label=None):
        if label is not None:
            if label in self.data:
                self.data[label] = []
        else:
            self.data = {}

    
    def get_values(self, label):
        return self.data[label]
    

    def summary(self):
        for label in self.data:
            avg = self.average(label)
            print(f"{label} - Average: {avg:.4f}")





class TrainingManager:
    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        epochs=10,
        device="cpu",
        trainstep_checkin_interval=100,
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.trainstep_checkin_interval = trainstep_checkin_interval
        self.tracker = ValueTracker()
        init_logger()

    def hasnan(self):
        for _, param in self.model.named_parameters():
            if torch.isnan(param).any():
                return True
        for _, param in self.model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                return True

        return False

    def trainstep(self, data):
        x, _ = data  # VAE only needs images, not labels
        x = x.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        recon_x, mu, logvar = self.model(x)
        
        # Calculate loss
        loss = self.model.loss_function(recon_x, x, mu, logvar)
        
        loss.backward()
        self.optimizer.step()
        
        self.tracker.add("Loss/trainstep", loss.item())
        self.tracker.add("Loss/epoch", loss.item())

    @torch.no_grad()
    def valstep(self, data):
        x, _ = data
        x = x.to(self.device)
        
        recon_x, mu, logvar = self.model(x)
        
        loss = self.model.loss_function(recon_x, x, mu, logvar)
        
        self.tracker.add("Loss/valstep", loss.item())
        self.tracker.add("Loss/val/epoch", loss.item())

    def on_trainloop_checkin(self, epoch, step, total_steps):
        avg_loss = self.tracker.average("Loss/trainstep")
        log_data({"Loss/train": avg_loss}, step + epoch * total_steps)
        self.tracker.reset("Loss/trainstep")

    def on_epoch_checkin(self, epoch):
        # Log training metrics
        train_loss = self.tracker.average("Loss/epoch")
        log_data({"Loss/epoch": train_loss}, epoch)
        self.tracker.reset("Loss/epoch")

        # Log validation metrics
        val_loss = self.tracker.average("Loss/val/epoch")
        log_data({"Loss/val/epoch": val_loss}, epoch)
        self.tracker.reset("Loss/val/epoch")

        # fancy image logging
        with torch.no_grad():
            test_batch, _ = next(iter(self.dataloader))
            test_batch = test_batch[:8].to(self.device)# first 8
            
            recon_batch, _, _ = self.model(test_batch)
            
            comparison = torch.cat([test_batch, recon_batch])
            log_img("reconstructions", comparison, epoch)

    def _save(self, name="latest.pt"):
        with open(os.path.join(self.dir, "ckpt", name), "wb+") as f:
            torch.save(self.model.state_dict(), f)

    def _load(self, name="latest.pt"):
        self.model.load_state_dict(
            torch.load(os.path.join(self.dir, "ckpt", name), weights_only=True)
        )

    def write_resume(self, epoch):
        with open(os.path.join(self.dir, "ckpt", "resume.txt"), "w+") as f:
            f.write(str(epoch))

    def get_resume(self):
        try:
            with open(os.path.join(self.dir, "ckpt", "resume.txt"), "r") as f:
                return int(f.read())
        except (FileNotFoundError, ValueError):
            return 0

    def write_best_val_loss(self, loss):
        with open(os.path.join(self.dir, "ckpt", "best_val_loss.txt"), "w+") as f:
            f.write(f"{loss:.6f}")

    def get_best_val_loss(self):
        try:
            with open(os.path.join(self.dir, "ckpt", "best_val_loss.txt"), "r") as f:
                return float(f.read())
        except (FileNotFoundError, ValueError):
            return float("inf")

    def resume(self):
        self._load("latest.pt")

    def save(self, loss):
        self._save("latest.pt")

        best_val_loss = self.get_best_val_loss()
        if loss < best_val_loss:
            best_val_loss = loss
            self._save("best.pt")
            self.write_best_val_loss(best_val_loss)

        # self._save(f"{prefix}_{step}.pt")

    def epoch(self, epoch: int, dataloader, val_loader=None):
        for step, data in enumerate(tqdm(dataloader, leave=False, dynamic_ncols=True)):
            self.trainstep(data)

            if (
                step % self.trainstep_checkin_interval
                == self.trainstep_checkin_interval - 1
            ):
                self.on_trainloop_checkin(epoch, step, len(dataloader))

        if val_loader is not None:
            for step, data in enumerate(
                tqdm(val_loader, leave=False, dynamic_ncols=True)
            ):
                self.valstep(data)

        self.on_epoch_checkin(epoch)

    def train(self, epochs=None, dataloader=None):

        if epochs is not None:
            self.epochs = epochs

        if dataloader is not None:
            self.dataloader = dataloader

        for e in trange(self.epochs, dynamic_ncols=True):

            if e <= self.resume_amt:
                continue

            self.epoch(e, self.dataloader, self.val_dataloader)