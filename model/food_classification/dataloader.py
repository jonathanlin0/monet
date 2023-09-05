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


# create a json file for prep

CIFAR_10_DIR = "model/datasets/cifar100"
FOOD_11_DIR = "model/datasets/food11"
FOOD_101_DIR = "model/datasets/food101/images"

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
                    file_path = f"{CIFAR_10_DIR}/{split}/{category}/{subcategory}/{file}"
                    if category == "fruit_and_vegetables":
                        correct_dict.append([file_path, 1])
                    else:
                        correct_dict.append([file_path, 0])

# entire dataset is food
def prep_food11():
    dataset_structure = {
        "/training": train,
        "/evaluation": val,
        "/validation": val
    }
    for split in dataset_structure:
        correct_dict = dataset_structure[split]
        categories = os.listdir(FOOD_11_DIR + split)
        for category in categories:
            if "." in category:
                continue
            for file in os.listdir(FOOD_11_DIR + split + "/" + category):
                file_path = f"{FOOD_11_DIR}{split}/{category}/{file}"
                correct_dict.append([file_path, 1])
            
# entire dataset is food
def prep_food101():
    categories = os.listdir(FOOD_101_DIR)
    for category in categories:
        if "." in category:
            continue
        for file in os.listdir(FOOD_101_DIR + "/" + category):
            file_path = f"{FOOD_101_DIR}/{category}/{file}"
            train.append([file_path, 1])

# PREPARATION FUNCTIONS
prep_cifar100()
prep_food11()
prep_food101()

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

        # if image is extremely long, cut the image left and right sides
        # if the image is extremely tall, cut the image top and bottom sides
        curr_width = image.size[0]
        curr_height = image.size[1]
        width_ratio = self.width / curr_width
        height_ratio = self.height / curr_height
        # shrink the image so that the dimension that has a lower ratio is the same as the target dimension
        if curr_width >= self.width and curr_height >= self.height:
            if width_ratio > height_ratio:
                shrink_ratio = height_ratio
                
                # resize image while maintaining ratio
                target_width = round(curr_width * shrink_ratio)
                target_height = round(curr_height * shrink_ratio)
                image = image.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)

                # cut off the edges of width
                curr_width = image.size[0]
                left_cutoff = floor((curr_width - self.width) / 2)
                right_cutoff = ceil((curr_width - self.width) / 2)
                image = image.crop((left_cutoff, 0, curr_width - right_cutoff, self.height))
            else:
                shrink_ratio = width_ratio

                # resize image while maintaining ratio
                target_width = round(curr_width * shrink_ratio)
                target_height = round(curr_height * shrink_ratio)
                image = image.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)

                # cut off the edges of height
                curr_height = image.size[1]
                top_cutoff = floor((curr_height - self.height) / 2)
                bottom_cutoff = ceil((curr_height - self.height) / 2)
                image = image.crop((0, top_cutoff, self.width, curr_height - bottom_cutoff))
        # one of the dimensions is too small
        # scale the image (keeping aspect ratio) so that the smaller dimension is the same as the target dimension
        elif height_ratio > width_ratio:
            # upscale for height to match
            scale_ratio = self.height / curr_height
            target_width = round(curr_width * scale_ratio)
            target_height = round(curr_height * scale_ratio)
            image = image.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)

            # cut off the edges of width
            curr_width = image.size[0]
            left_cutoff = floor((curr_width - self.width) / 2)
            right_cutoff = ceil((curr_width - self.width) / 2)
            image = image.crop((left_cutoff, 0, curr_width - right_cutoff, self.height))
        else:
            # upscale for width to match
            scale_ratio = self.width / curr_width
            target_width = round(curr_width * scale_ratio)
            target_height = round(curr_height * scale_ratio)
            image = image.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)

            # cut off the edges of height
            curr_height = image.size[1]
            top_cutoff = floor((curr_height - self.height) / 2)
            bottom_cutoff = ceil((curr_height - self.height) / 2)
            image = image.crop((0, top_cutoff, self.width, curr_height - bottom_cutoff))
        
        
        if self.transform is not None:
            image = self.transform(image)

        image = image.to(torch.float32)

        # pad the image

        label = torch.tensor(self.data_dict[idx][1])

        return (image, label)

def get_data(batch_size=32, num_workers=8, height=500, width=500):
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