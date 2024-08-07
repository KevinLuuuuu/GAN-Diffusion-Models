import torch
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import csv
from argparse import ArgumentParser, Namespace
from pathlib import Path
from p3_model import DANN
from p2_modules import ImageDataset
import random
import numpy as np

def main(args):

    dataset_path = args.input_dir #"./hw1_data/p1_data/val_50"
    output_path = args.output_dir #"./p1_output.csv"


    if "svhn" in str(dataset_path):
        target_dataset_name = "svhn"
        ckpt_path = "p3_svhn.pth" ##################改
    elif "usps" in str(dataset_path):
        target_dataset_name = "usps"
        ckpt_path = "p3_usps.pth" ##################改
    
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

    if target_dataset_name == 'svhn':
        test_set = ImageDataset(str(dataset_path), transform=test_transform3, train_set = False)

    elif target_dataset_name == 'usps':
        test_set = ImageDataset(str(dataset_path), transform=test_transform1, train_set = False)
    print(len(test_set))
        
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    pred_label_list = []
    image_name_list = []

    with torch.no_grad():
        for i, (image, image_name) in enumerate(tqdm(test_loader)):
            image = image.to(device)
            class_out, _, _ = model(image, 0)
            pred_label = torch.max(class_out.data, 1)[1]
            pred_label_list.append(pred_label)
            image_name_list.append(image_name)
    print(len(image_name_list))

    ################# check ####################
    with open(output_path, 'w', newline="") as fp:        
        file_writer = csv.writer(fp)
        file_writer.writerow(['image_name', 'label'])
        for i in range(len(pred_label_list)):
            for j in range(len(pred_label_list[i])):
                file_writer.writerow([image_name_list[i][j], pred_label_list[i][j].item()])

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        help="Path to the input file.",
        required=True
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to the output file.",
        required=True
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu" 
    #print(device)
    torch.cuda.empty_cache()

    args = parse_args()
    main(args)
                 