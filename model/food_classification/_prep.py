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
from tqdm import tqdm
import random


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
    val_split_prob = 0.2
    categories = os.listdir(FOOD_101_DIR)
    for category in categories:
        if "." in category:
            continue
        for file in os.listdir(FOOD_101_DIR + "/" + category):
            file_path = f"{FOOD_101_DIR}/{category}/{file}"
            if random.random() <= val_split_prob:
                val.append([file_path, 1])
            else:
                train.append([file_path, 1])

# PREPARATION FUNCTIONS
prep_cifar100()
prep_food11()
prep_food101()

# delete all contents in cache_photos
try:
    shutil.rmtree('model/cache_photos')
except:
    pass
os.mkdir("model/cache_photos")

dicts = [train, val]

height = 32
width = 32
for dict_ in dicts:
    for i, key in enumerate(tqdm(dict_)):
        image_path = key[0]
        image = PIL.Image.open(image_path, mode="r")
        # if image is extremely long, cut the image left and right sides
        # if the image is extremely tall, cut the image top and bottom sides
        curr_width = image.size[0]
        curr_height = image.size[1]
        width_ratio = width / curr_width
        height_ratio = height / curr_height
        # shrink the image so that the dimension that has a lower ratio is the same as the target dimension
        if curr_width >= width and curr_height >= height:
            if width_ratio > height_ratio:
                shrink_ratio = height_ratio
                
                # resize image while maintaining ratio
                target_width = round(curr_width * shrink_ratio)
                target_height = round(curr_height * shrink_ratio)
                image = image.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)

                # cut off the edges of width
                curr_width = image.size[0]
                left_cutoff = floor((curr_width - width) / 2)
                right_cutoff = ceil((curr_width - width) / 2)
                image = image.crop((left_cutoff, 0, curr_width - right_cutoff, height))
            else:
                shrink_ratio = width_ratio

                # resize image while maintaining ratio
                target_width = round(curr_width * shrink_ratio)
                target_height = round(curr_height * shrink_ratio)
                image = image.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)

                # cut off the edges of height
                curr_height = image.size[1]
                top_cutoff = floor((curr_height - height) / 2)
                bottom_cutoff = ceil((curr_height - height) / 2)
                image = image.crop((0, top_cutoff, width, curr_height - bottom_cutoff))
        # one of the dimensions is too small
        # scale the image (keeping aspect ratio) so that the smaller dimension is the same as the target dimension
        elif height_ratio > width_ratio:
            # upscale for height to match
            scale_ratio = height / curr_height
            target_width = round(curr_width * scale_ratio)
            target_height = round(curr_height * scale_ratio)
            image = image.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)

            # cut off the edges of width
            curr_width = image.size[0]
            left_cutoff = floor((curr_width - width) / 2)
            right_cutoff = ceil((curr_width - width) / 2)
            image = image.crop((left_cutoff, 0, curr_width - right_cutoff, height))
        else:
            # upscale for width to match
            scale_ratio = width / curr_width
            target_width = round(curr_width * scale_ratio)
            target_height = round(curr_height * scale_ratio)
            image = image.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)

            # cut off the edges of height
            curr_height = image.size[1]
            top_cutoff = floor((curr_height - height) / 2)
            bottom_cutoff = ceil((curr_height - height) / 2)
            image = image.crop((0, top_cutoff, width, curr_height - bottom_cutoff))
        
        image_path = image_path.replace("/", "_")
        image_path = image_path[:image_path.rfind(".")] # get rid of current file extension
        image_path = "model/cache_photos/" + image_path + ".png"
        image.save(image_path ,"PNG")
        dict_[i][0] = image_path

# remove all photos that are black and white (only working in RGB)
print("Removing grayscale photos...")
dicts = [train, val]
new_train = []
new_val = []
matching_dicts = [new_train, new_val]
for i, dict_ in enumerate(dicts):
    for key in tqdm(dict_):
        img_path = key[0]
        image = PIL.Image.open(img_path, mode="r")
        image = torchvision.transforms.ToTensor()(image)
        image = image.to(torch.float32)
        if image.shape[0] == 1:
            continue
        matching_dicts[i].append(key)
train = new_train
val = new_val

# save the json file
with open("model/food_classification/data.json", "w") as f:
    json.dump({"train": train, "val": val}, f, indent=4)