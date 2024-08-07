import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import torch.nn.parallel
from pytorch_fid import fid_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import math
import random
import numpy as np
from tqdm import tqdm
from p1 import FilterableImageFolder
from p1_b_model import Generator, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
seed = 5555
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)

batch_size = 64
transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

image_dir = "test_p1"
dataroot = 'hw2_data/face'

num_epochs = 150

# Data
trainset = FilterableImageFolder(root=dataroot,
                        valid_classes=['train'],
                        transform=transform
                        )
train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

netG = Generator(batch_size, 64, 128, 64).to(device)
netD = Discriminator(batch_size, 64, 64).to(device)

g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()), 0.0004, [0.5, 0.999])
d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), 0.0002, [0.5, 0.999])


fixed_noise = torch.randn(1000, 128, device=device)

G_losses = []
D_losses = []
best_fid_score = math.inf
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    #Training loop
    netD.train()
    netG.train()
    train_loss = []
    correct = 0
    for i, input in enumerate(tqdm(train_dataloader, 0)):
        input = input[0]
        b_size = input.size(0)
        input, label = input.to(device), torch.full((b_size, ), 1, device=device).to(torch.float32)
        d_optimizer.zero_grad()
        d_out_real,dr1,dr2 = netD(input)
        d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

        z = torch.randn(b_size, 128, device=device)
        fake_images,gf1,gf2 = netG(z)
        d_out_fake,df1,df2 = netD(fake_images)
        d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        g_optimizer.zero_grad()
        z = torch.randn(b_size, 128, device=device)
        fake_images,_,_ = netG(z)
        g_out_fake,_,_ = netD(fake_images)
        g_loss_fake = - g_out_fake.mean()
        g_loss_fake.backward()
        g_optimizer.step()

        D_losses.append(d_loss.item())
        G_losses.append(g_loss_fake.item())


    netG.eval()
    with torch.no_grad():
        fake, _, _ = netG(fixed_noise[:500])
        fake = ((fake.data + 1) / 2).clamp_(0, 1)
        for i in range(500):
            vutils.save_image(fake[i], str(image_dir) + "/" + str(i) + ".png")
        fake, _, _ = netG(fixed_noise[500:])
        fake = ((fake.data + 1) / 2).clamp_(0, 1)
        for i in range(500):
            vutils.save_image(fake[i], str(image_dir) + "/" + str(i+500) + ".png")
    output = os.popen('python -m pytorch_fid hw2_data/face/val p1_image/').read()
    fid_score = float(output.replace('FID:', ''))
    print("Epoch", epoch)
    print("Gloss : ", sum(G_losses)/len(G_losses))
    print("Dloss : ", sum(D_losses)/len(D_losses))
    print("FID Score: {:.3f}".format(fid_score))
    if fid_score<best_fid_score:
        best_fid_score = fid_score
        torch.save(netD, 'netD.pth')
        torch.save(netG, 'netG.pth')
        print("Best fid_score in this epoch and save model.")
    print("Best fid_score is {:.3f}".format(best_fid_score))
print(best_fid_score)