import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import torchvision
import PIL

# create a json file for prep

CIFAR_10_DIR = "model/datasets/cifar100"

train = []
val = []

def prep_cifar100():
    # go through each category and subcategory
    # if the category is fruit_and_vegetables, then mark the value as 1. else, mark it as false

    # get the list of the files in cifar_10_dir

    # key is the name of the folder, value is the data split the values in the folder will be designated to
    dataset_structure = {
        "/train": train,
        "/test": val
    }
    for split in dataset_structure:
        correct_dict = dataset_structure[split]
        categories = os.listdir(CIFAR_10_DIR + split)
        for category in categories:
            if "." in category: # for the .DS_Store file
                continue
            for subcategory in os.listdir(CIFAR_10_DIR + split + "/" + category):
                if "." in subcategory:  # for the .DS_Store file
                    continue
                for file in os.listdir(CIFAR_10_DIR + split + "/" + category + "/" + subcategory):
                    if category == "fruit_and_vegetables":
                        correct_dict.append([file, split, category, subcategory, 1])
                    else:
                        correct_dict.append([file, split, category, subcategory, 0])

# PREPARATION FUNCTIONS
prep_cifar100()

train = train[:max(len(train), 1000)]
val = val[:max(len(val), 1000)]

# DATASET AND DATALOADER -----------------------------------------
class food_classification_dataset(Dataset):
    """dataset for Animal Kingdom"""

    def __init__(self, data_dict, root_dir, transform=None):
        self.data_dict = data_dict
        self.root_dir = root_dir
        self.transform = transform
        

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        # this dictionary converts the string labels to integers
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        split = self.data_dict[idx][1]
        category = self.data_dict[idx][2]
        subcategory = self.data_dict[idx][3]

        image = PIL.Image.open(f"{self.root_dir}/{split}/{category}/{subcategory}/{self.data_dict[idx][0]}", mode="r")
        # have to use PIL instead of io.imread because transform expects PIL image
        # image = io.imread(img_name)
        
        if self.transform is not None:
            image = self.transform(image)

        image = image.to(torch.float32)
        label = torch.tensor(self.data_dict[idx][4])

        return (image, label)

def get_data(batch_size=32, num_workers=8):
    # cwd = os.path.dirname(os.path.realpath(__file__))
    # cwd = cwd[0:cwd.rfind("/")]
    # cwd = cwd[0:cwd.rfind("/") + 1]
    # set the root directory of the project to 2 layers above the current dataloader


    train_dataset = food_classification_dataset(
        root_dir = "model/datasets/cifar100",
        data_dict = train,
        transform=torchvision.transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandAugment(),
                        transforms.ToTensor()
                     ])
    )

    val_dataset = food_classification_dataset(
        root_dir = "model/datasets/cifar100",
        data_dict = val,
        transform=torchvision.transforms.Compose([
                        transforms.ToTensor()
                     ])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = False
    )

    return train_loader, val_loader