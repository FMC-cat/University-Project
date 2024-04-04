import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from cyclegan.discriminator_model import Discriminator
from cyclegan.generator_model import Generator
import cyclegan.config as config
from cyclegan.utils import save_checkpoint, load_checkpoint
from cyclegan.config import load_model,get_dir
from cyclegan.dataset import HorseZebraDataset

from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
def test_fn(gen_Z, gen_H, loader,type_name):
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)
        fake_zebra = gen_Z(horse)
        
        t_fake_zebra = (fake_zebra*0.5+0.5).data.cpu().numpy()[0].swapaxes(0, 1).swapaxes(1, 2)
        t_fake_zebra = t_fake_zebra*255

    return t_fake_zebra

def run(type_name , img):
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            load_model(type_name)[0],
            gen_H,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            load_model(type_name)[1],
            gen_Z,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            load_model(type_name)[2],
            disc_H,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            load_model(type_name)[3],
            disc_Z,
            opt_disc,
            config.LEARNING_RATE,
        )
    
    val_dataset = HorseZebraDataset(
        root_horse=img,
        root_zebra=get_dir(type_name)[1],
        transform=config.transforms,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    return test_fn(gen_Z,gen_H,val_loader,type_name)

