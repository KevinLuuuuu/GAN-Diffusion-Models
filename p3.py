import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import transforms
import numpy as np
from p2_modules import ImageDataset
from torch.utils.data import DataLoader
from p3_model import DANN
from tqdm.auto import tqdm
import math

source_dataset_name = 'mnistm'
target_dataset_name = 'svhn' # svhn, usps
train_method = "target" # source, dann, target

source_image_root = os.path.join("hw2_data/digits", source_dataset_name)
target_image_root = os.path.join("hw2_data/digits", target_dataset_name)

source_train_csv = os.path.join("hw2_data/digits", source_dataset_name, "train.csv")
source_test_csv = os.path.join("hw2_data/digits", source_dataset_name, "val.csv")
target_train_csv = os.path.join("hw2_data/digits", target_dataset_name, "train.csv")
target_test_csv = os.path.join("hw2_data/digits", target_dataset_name, "val.csv")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU:', device)

cudnn.benchmark = True

lr = 1e-3
batch_size = 128
image_size = 28
n_epoch = 100

seed = 5203
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

# load data

transorms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

transorms_one = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

source_train_dataset = ImageDataset(source_image_root, transform=transorms, csv_path=source_train_csv)
source_train_dataloader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True)

source_test_dataset = ImageDataset(source_image_root, transform=transorms, csv_path=source_test_csv)
source_test_dataloader = DataLoader(source_test_dataset, batch_size=batch_size, shuffle=False)

if target_dataset_name == 'svhn':
    target_train_dataset = ImageDataset(target_image_root, transform=transorms, csv_path=target_train_csv)
    target_train_dataloader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True)
    target_test_dataset = ImageDataset(target_image_root, transform=transorms, csv_path=target_test_csv)
    target_test_dataloader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False)

elif target_dataset_name == 'usps':
    target_train_dataset = ImageDataset(target_image_root, transform=transorms_one, csv_path=target_train_csv)
    target_train_dataloader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True)
    target_test_dataset = ImageDataset(target_image_root, transform=transorms_one, csv_path=target_test_csv)
    target_test_dataloader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False)

# load model

model = DANN().to(device)
print(model)

# setup optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss().to(device)
best_loss = math.inf

# training

for epoch in range(n_epoch):

    model.train()
    train_loss = 0
    train_loss_record = []
    train_correct = 0

    len_dataloader = min(len(source_train_dataloader), len(target_train_dataloader))
    data_source_iter = iter(source_train_dataloader)
    data_target_iter = iter(target_train_dataloader)

    i = 0
    while i < len_dataloader:
        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        data_source = data_source_iter.next()
        data_target = data_target_iter.next()

        s_img, s_label = data_source
        s_img = s_img.to(device)
        s_label = s_label.to(device)
        t_img, t_label = data_target
        t_img = t_img.to(device)
        t_label = t_label.to(device)

        batch_size_s = len(s_img)
        batch_size_t = len(t_img)

        model.zero_grad()

        #print('#', end="")
        if train_method == "source":
            # training model using source data
            class_output, _= model(x=s_img, alpha=alpha)
            loss_s = criterion(class_output, s_label)
            err = loss_s
            train_loss_record.append(err.item())
            err.backward()
            optimizer.step()

        elif train_method == "dann":
            # training model using source data
            domain_label = torch.zeros(batch_size_s)
            domain_label = domain_label.long()
            domain_label = domain_label.to(device)

            class_output, domain_output,_ = model(x=s_img, alpha=alpha)
            err_s_label = criterion(class_output, s_label)
            err_s_domain = criterion(domain_output, domain_label)

            # training model using target data
            
            domain_label = torch.ones(batch_size_t)
            domain_label = domain_label.long()
            domain_label = domain_label.to(device)

            _, domain_output,_ = model(x=t_img, alpha=alpha)
            err_t_domain = criterion(domain_output, domain_label)

            err = err_t_domain + err_s_domain + err_s_label
            train_loss_record.append(err.item())
            err.backward()
            optimizer.step()

        else:
            class_output, _,_= model(x=t_img, alpha=alpha)
            loss_t = criterion(class_output, t_label)
            err = loss_t
            train_loss_record.append(err.item())
            err.backward()
            optimizer.step()

        i += 1
    
    #print()
    model.eval()
    eval_loss = 0
    eval_loss_record_s = []
    eval_loss_record_t = []
    eval_correct_s = 0
    eval_correct_t = 0

    with torch.no_grad():
        for i, (image, label) in enumerate(tqdm(source_test_dataloader)):
            image, label = image.to(device), label.to(device)
            class_output,_,_ = model(image, alpha)
            eval_loss = criterion(class_output, label)
            eval_loss_record_s.append(eval_loss.item())
            pred_label = torch.max(class_output.data, 1)[1]
            eval_correct_s = eval_correct_s + pred_label.eq(label.view_as(pred_label)).sum().item()
            
        for i, (image, label) in enumerate(tqdm(target_test_dataloader)):
            image, label = image.to(device), label.to(device)
            class_output,_,_  = model(image, alpha)
            eval_loss = criterion(class_output, label)
            eval_loss_record_t.append(eval_loss.item())
            pred_label = torch.max(class_output.data, 1)[1]
            eval_correct_t = eval_correct_t + pred_label.eq(label.view_as(pred_label)).sum().item()
        
    
    mean_train_loss = sum(train_loss_record)/len(train_loss_record)
    valid_acc_s = 100 * eval_correct_s / len(source_test_dataset)
    mean_eval_loss_s = sum(eval_loss_record_s)/len(eval_loss_record_s)
    valid_acc_t = 100 * eval_correct_t / len(target_test_dataset)
    mean_eval_loss_t = sum(eval_loss_record_t)/len(eval_loss_record_t)

    print("Epoch:", epoch)
    print('Train loss: {:.4f}'.format(err))
    print('Evaluate loss on source: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(mean_eval_loss_s, eval_correct_s, len(source_test_dataset), valid_acc_s))     
    print('Evaluate loss on target: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(mean_eval_loss_t, eval_correct_t, len(target_test_dataset), valid_acc_t))    

    if train_method == "dann":
        total_loss = mean_eval_loss_s + mean_eval_loss_t
    else:
        total_loss = mean_eval_loss_t

    if total_loss < best_loss:
        best_loss = total_loss
        print('This epoch has best loss is {:.4f} and save model'.format(best_loss))
        torch.save(model.state_dict(), "p3_model/p3_6.pth")

    print('The best loss is {:.0f} '.format(best_loss))

