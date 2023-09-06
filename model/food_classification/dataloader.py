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
from PIL import Image
from math import floor
from math import ceil
from statistics import median
import shutil


# create a json file for prep

CIFAR_10_DIR = "model/datasets/cifar100"
FOOD_11_DIR = "model/datasets/food11"
FOOD_101_DIR = "model/datasets/food101/images"


# graph stuff
# max_width = 0
# max_height = 0
# min_width = 100000
# min_height = 100000
# avg_height = []
# avg_width = []
# for key in train + val:
#     image_path = key[0]

#     # Open an image
#     image = Image.open(image_path)

#     # Get the dimensions (width and height)
#     width, height = image.size
#     max_width = max(max_width, width)
#     max_height = max(max_height, height)
#     min_width = min(min_width, width)
#     min_height = min(min_height, height)
#     avg_width.append(width)
#     avg_height.append(height)

# print(max_width)
# print(max_height)
# print(min_width)
# print(min_height)
# print(median(avg_width))
# print(median(avg_height))

# # plot graph in matplotlib where the x axis is the width and the y axis is the count
# # set max y axis value to 1000
# plt.hist(avg_width, bins = 100)
# plt.show()
# exit()

import json
f = open("model/food_classification/data.json", "r")
data = json.load(f)
train = data["train"]
val = data["val"]
f.close()


# DATASET AND DATALOADER -----------------------------------------
class food_classification_dataset(Dataset):
    """dataset for Animal Kingdom"""

    def __init__(self, data_dict, root_dir, width, height, transform=None):
        self.data_dict = data_dict
        self.root_dir = root_dir
        self.transform = transform
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        # this dictionary converts the string labels to integers
        
        if torch.is_tensor(idx):
            idx = idx.tolist()


        image = PIL.Image.open(self.data_dict[idx][0], mode="r")
        # have to use PIL instead of io.imread because transform expects PIL image
        # image = io.imread(img_name)
        
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torchvision.transforms.ToTensor()(image)

        image = image.to(torch.float32)

        label = torch.tensor(self.data_dict[idx][1])

        return (image, label)

def get_data(batch_size=8, num_workers=4, height=100, width=100):
    # cwd = os.path.dirname(os.path.realpath(__file__))
    # cwd = cwd[0:cwd.rfind("/")]
    # cwd = cwd[0:cwd.rfind("/") + 1]
    # set the root directory of the project to 2 layers above the current dataloader


    train_dataset = food_classification_dataset(
        root_dir = "model/datasets/cifar100",
        data_dict = train,
        width = width,
        height = height,
        transform=torchvision.transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandAugment(),
                        transforms.ToTensor()
                     ])
    )

    val_dataset = food_classification_dataset(
        root_dir = "model/datasets/cifar100",
        data_dict = val,
        width = width,
        height = height,
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