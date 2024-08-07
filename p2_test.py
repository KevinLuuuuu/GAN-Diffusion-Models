import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from p2_utils import *
from p2_modules import UNet_conditional, EMA
from p2 import Diffusion
from argparse import ArgumentParser, Namespace
from pathlib import Path
import random

def main(args):

    image_dir = args.output_dir

    ckpt_path = 'p2.pth'

    # CUDA
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU State:', device)

    # Random seed
    seed = 100
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    model = UNet_conditional(num_classes=10).to(device)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    diffusion = Diffusion(noise_steps=400, img_size=28, device=device)    
    
    number = 100
    for i in range(10):
        labels = torch.full((number, ), i, dtype=torch.long, device=device)
        sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
        for j in range(number):
            name = str(j+1)
            name = name.zfill(3)
            save_images(sampled_images[j], str(image_dir) + "/" + str(i) + "_" + name + ".png")

    #labels = torch.arange(10).long().to(device)
    #sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
    #ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
    #for i in range(9):
    #sampled_images = torch.cat(sampled_images, (diffusion.sample(model, n=len(labels), labels=labels)))
    #ema_sampled_images = torch.cat(ema_sampled_images, (diffusion.sample(model, n=len(labels), labels=labels)))
    
    
    #diffusion = Diffusion(noise_steps=1000, img_size=28, device=device)
    
    '''
    rep = []
    # draw 100 images
    labels = torch.full((10, ), 0, dtype=torch.long, device=device)
    #sampled_images0 = diffusion.sample(model, n=len(labels), labels=labels)
    #save_images(sampled_images0, "7878.png")
    for i in range(1, 10):
        labels = torch.cat((labels, torch.full((10, ), i, dtype=torch.long, device=device)), 0)
    sampled_images, x = diffusion.sample(model, n=len(labels), labels=labels)

    save_images(x, "report.png")
    save_images(sampled_images[0][0], "report0.png")
    save_images(sampled_images[1][0], "report200.png")
    save_images(sampled_images[2][0], "report400.png")
    save_images(sampled_images[3][0], "report600.png")
    save_images(sampled_images[4][0], "report800.png")
    save_images(sampled_images[5][0], "report900.png")
    '''
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to the input file.",
        required=True
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    torch.cuda.empty_cache()
    args = parse_args()
    main(args)
