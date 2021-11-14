# Adapted from Lucidrains' examples
from model import Model
from diffusion_wrapper import DiffusionWrapper

import tqdm
import time
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from torchvision.datasets import STL10
from torch.utils.data import DataLoader

#constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 64
GRADIENT_ACCUMULATE_EVERY = 4
LR = 1e-3
VALIDATION_EVERY = 100
GENERATE_EVERY = 500


#helpers
def cycle(loader):
    while True:
        for data in loader:
            yield data

def scale(x):
    return x * 2 - 1

def rescale(x):
    return (x + 1) / 2

def train():
    wandb.init(project="stl10-diffusion-model") #copilot chose this project name
    model = DiffusionWrapper(Model(), input_shape=(3, 96, 96))
    model.cuda()

    train_dataset = STL10(root='./data', split='train', download=True, transform=transforms.ToTensor(), download=True)
    val_dataset = STL10(root='./data', split='test', download=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    #optimizer
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    #training
    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc="training"):
        start_time = time.time()
        model.train()

        for _ in range(GRADIENT_ACCUMULATE_EVERY):
            batch, _ = next(train_loader)
            loss = Model(scale(batch))
            loss.backward()
        
        end_time = time.time()
        print(f'train loss: {loss.item()}')
        train_loss = loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        if i % VALIDATION_EVERY == 0:
            model.eval()
            with torch.no_grad():
                batch, _ = next(val_loader)
                loss = model(scale(batch))
                print(f'val loss: {loss.item()}')
                val_loss = loss.item()
        
        if i % GENERATE_EVERY == 0:
            model.eval()
            samples = model.generate(1)
            image_array = rescale(samples)
            images = wandb.Image(image_array, step=i, caption="Generated")
            wandb.log({"Generated": images}, step=i)
        
        logs = {}
        logs = {
            **logs,
            'iter': i,
            'time': end_time - start_time,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        wandb.log(logs, step=i)
    
    wandb.finish()


if __name__ == "__main__":
    train()
