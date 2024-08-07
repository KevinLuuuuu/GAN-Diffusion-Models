import torch
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import torchvision.models as models
from types import SimpleNamespace
import csv
from argparse import ArgumentParser, Namespace
from pathlib import Path
from p3_model import DANN
from p2_modules import ImageDataset
import random
import numpy as np

def main():

    seed = 5203
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    ckpt_path = "p3_model/p3_6.pth"
    target_dataset_name = "usps"

    target_image_root = os.path.join("hw2_data/digits", target_dataset_name)
    target_test_csv = os.path.join("hw2_data/digits", target_dataset_name, "val.csv")

    
    model = DANN().to(device)

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    test_transform1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean= 0.5, std=0.5)
    ])

    batch_size = 10
    test_set = ImageDataset(target_image_root, transform=test_transform1, csv_path=target_test_csv)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    eval_correct_t = 0

    for i, (image, label) in enumerate(tqdm(test_loader)):
        image, label = image.to(device), label.to(device)
        class_output,_  = model(image, 0)
        pred_label = torch.max(class_output.data, 1)[1]
        eval_correct_t = eval_correct_t + pred_label.eq(label.view_as(pred_label)).sum().item()
    
    valid_acc_t = 100 * eval_correct_t / len(test_set)
    print(valid_acc_t)
            


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu" 
    torch.cuda.empty_cache()
    main()

                 