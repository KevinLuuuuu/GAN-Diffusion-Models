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
from sklearn.manifold import TSNE
import random
import matplotlib.pyplot as plt
import numpy as np

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu" 
#print(device)
torch.cuda.empty_cache()

ckpt_path = "p3_model/p3_5.pth"

model = DANN().to(device)

ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt)
model = model.to(device)
model.eval()

test_transform3 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

test_transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

batch_size = 10
representation, label_list, domain_list = [], [], []


###by domain
source_dataset_name = "mnistm"
source_test_csv = os.path.join("hw2_data/digits", source_dataset_name, "val.csv")
dataset_path = os.path.join("hw2_data/digits", source_dataset_name)
source_set = ImageDataset(dataset_path, transform=test_transform3, csv_path=source_test_csv)
test_loader = DataLoader(source_set, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    for i, (image, label) in enumerate(tqdm(test_loader)):
        image = image.to(device)
        _, _, repre = model(image, 0.1) #改這
        for rep, lab in zip(repre, label):
            rep = rep.reshape(-1)
            representation.append(rep.cpu().numpy())
            label_list.append(lab)
            domain_list.append(0)



target_dataset_name = "usps"
target_test_csv = os.path.join("hw2_data/digits", target_dataset_name, "val.csv")
dataset_path = os.path.join("hw2_data/digits", target_dataset_name)

if target_dataset_name == 'svhn':
    test_set = ImageDataset(dataset_path, transform=test_transform3, csv_path=target_test_csv)

elif target_dataset_name == 'usps':
    test_set = ImageDataset(dataset_path, transform=test_transform1, csv_path=target_test_csv)
    
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

with torch.no_grad():
    for i, (image, label) in enumerate(tqdm(test_loader)):
        image = image.to(device)
        _, _, repre = model(image, 0.1) #改這
        for rep, lab in zip(repre, label):
            rep = rep.reshape(-1)
            representation.append(rep.cpu().numpy())
            label_list.append(lab)
            domain_list.append(1)

rep_arr = np.array(representation)
lab_arr = np.array(label_list)
dom_arr = np.array(domain_list)

# prepare color
color = plt.get_cmap('gist_ncar')
colors = []
for i in range(10):
    colors.append(color(i/10))
plt.figure(figsize=(10, 10))

#colors = ['b', 'r']

# t-sne
XY_tsne = TSNE(n_components=2, init='random', random_state=5203).fit_transform(rep_arr)
XY_tsne_min = XY_tsne.min(0) 
XY_tsne_max = XY_tsne.max(0)
XY_norm = (XY_tsne - XY_tsne_min)/(XY_tsne_max - XY_tsne_min)

#print(XY_norm.shape)

for i in range(XY_norm.shape[0]):
    plt.plot(XY_norm[i, 0], XY_norm[i, 1], 'o', color=colors[lab_arr[i]])
plt.savefig('TSNE_usps_class.png')


# prepare color
colors = ['b', 'r']
# t-sne
for i in range(XY_norm.shape[0]):
    plt.plot(XY_norm[i, 0], XY_norm[i, 1], 'o', color=colors[dom_arr[i]])
plt.savefig('TSNE_usps_domain.png')

