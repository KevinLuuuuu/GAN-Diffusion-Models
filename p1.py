import torch.nn as nn
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from tqdm import tqdm
from torchvision.datasets.folder import *
from typing import *
from face_recog import face_recog
import os
#from p1_b_model import Generator, Discriminator

class FilterableImageFolder(ImageFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            valid_classes: List = None
    ):
        self.valid_classes = valid_classes
        super(FilterableImageFolder, self).__init__(root, transform, target_transform, loader, is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        classes = [valid_class for valid_class in classes if valid_class in self.valid_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx



# Discriminator
class Discriminator(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(inputSize, hiddenSize, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hiddenSize, hiddenSize*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hiddenSize*2, hiddenSize*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hiddenSize*4, hiddenSize*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hiddenSize*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid())

    def forward(self, input):
        return self.main(input)

# Generator
class Generator(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(inputSize, hiddenSize*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hiddenSize*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(hiddenSize*8, hiddenSize*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(hiddenSize*4, hiddenSize*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(hiddenSize*2, hiddenSize, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddenSize),
            nn.ReLU(True),

            nn.ConvTranspose2d(hiddenSize, outputSize, 4, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, input):
        return self.main(input)

# CUDA
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

# Random seed
manualSeed = 5555
print('Random Seed:', manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Attributes
dataroot = 'hw2_data/face'

batch_size = 16
image_size = 64
G_out_D_in = 3
G_in = 100
G_hidden = 64  
D_hidden = 64

epochs = 250
g_lr = 0.002
d_lr = 0.002
beta1 = 0.5
img_list = []

image_dir = "p1_image"

# Data
dataset = FilterableImageFolder(root=dataroot,
                           valid_classes=['train'],
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))

# Create the dataLoader
dataLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Weights
def weights_init(m):
    classname = m.__class__.__name__
    #print('classname:', classname)

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Train
def train():
    # Create the generator
    netG = Generator(G_in, G_hidden, G_out_D_in).to(device)
    netG.apply(weights_init)
    print(netG)

    # Create the discriminator
    netD = Discriminator(G_out_D_in, D_hidden).to(device)
    netD.apply(weights_init)
    print(netD)

    # Loss fuG_out_D_intion
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(1000, G_in, 1, 1, device=device)

    real_label = 1
    fake_label = 0
    optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(beta1, 0.999), weight_decay=0.0003)
    optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(beta1, 0.999))

    #img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    best_acc = 0
    best_fid_score = 1000
    print('Start!')
    print(len(dataLoader))
    for epoch in range(epochs):
        for i, data in enumerate(tqdm(dataLoader, 0)):
            # Update D network
            #print((data))
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device).to(torch.float32)
            output,df1,df2 = netD(real_cpu)

            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, G_in, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)

            errD_fake = criterion(output, label)
            errD_fake.backward()

            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update G network
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1
            #if i==1:
                #break
        
        # test acc and save model
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        for i in range(1000):
            vutils.save_image(fake[i], image_dir + "/" + str(i) + ".png")
        acc = face_recog(image_dir)
        output = os.popen('python -m pytorch_fid hw2_data/face/val p1_image/').read()
        fid_score = float(output.replace('FID:', ''))
        print("Epoch", epoch)
        print("Gloss : ", sum(G_losses)/len(G_losses))
        print("Dloss : ", sum(D_losses)/len(D_losses))
        print("Face recognition Accuracy: {:.3f}%".format(acc))
        print("FID Score: {:.3f}".format(fid_score))
        if fid_score<best_fid_score:
            best_fid_score = fid_score
            torch.save(netD, 'netD.pkl')
            torch.save(netG, 'netG.pkl')
            print("Best fid_score in this epoch and save model.")
        print("Best fid_score is {:.3f}".format(best_fid_score))
        
    return G_losses, D_losses


# Plot
def plotImage(G_losses, D_losses):
    print('Start to plot!!')
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    #plt.show()
    plt.savefig("Loss.png")

if __name__ == '__main__':
    G_losses, D_losses = train()
    plotImage(G_losses, D_losses)

