import torch.nn as nn
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from face_recog import face_recog
from argparse import ArgumentParser, Namespace
from pathlib import Path
from p2_utils import save_images
from p1_b_model import Generator

def main(args):

    ckpt_path = 'p1.pth'

    # CUDA
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU State:', device)

    # Random seed
    #for i in range(1, 100):
    manualSeed = 12#12
    print('Random Seed:', manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.backends.cudnn.deterministic = True

    batch_size = 64

    netG = Generator(batch_size, 64, 128, 64).to(device)
    ckpt = torch.load(ckpt_path)
    netG.load_state_dict(ckpt)

    fixed_noise = torch.randn(1000, 128, 1, 1, device=device)
    image_dir = args.output_dir

    with torch.no_grad():
        fake,_,_ = netG(fixed_noise[0:500])
        fake = ((fake.data + 1) / 2).clamp_(0, 1)
        for i, a in enumerate(fake):
            vutils.save_image(fake[i], str(image_dir) + "/" + str(i) + ".png")
    with torch.no_grad():
        fake,_,_ = netG(fixed_noise[500:])
        fake = ((fake.data + 1) / 2).clamp_(0, 1)
        for i, a in enumerate(fake):
            vutils.save_image(fake[i], str(image_dir) + "/" + str(i+500) + ".png")

    acc = face_recog(str(image_dir))
    print("Face recognition Accuracy: {:.3f}%".format(acc))

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
