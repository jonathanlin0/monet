import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import torchvision
import tarfile
import numpy as np
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wandb

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from torchmetrics import Accuracy
import torch.utils.data as data
from torchvision.datasets import MNIST
from torch.nn import functional as F

parent_dir = os.path.abspath(__file__)[:os.path.abspath(__file__).rfind("/")]
parent_dir = parent_dir[:parent_dir.rfind("/")]
parent_dir = parent_dir[:parent_dir.rfind("/")]
sys.path.append(parent_dir)

from model.food_classification import dataloader

class food_classifier(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        dropout: float,
        num_classes: int,
        track_wandb:bool,
        img_channels=3,
        num_starting_filters=512,
        num_linear_layers=4,
        num_conv_layers=4):
        super().__init__()

        self.prep = nn.Conv2d(
                in_channels = img_channels,
                out_channels = num_starting_filters,
                kernel_size = 7,
                stride = 2,
                padding = 3,
                bias = False
        )
        self.norm_1 = nn.BatchNorm2d(num_starting_filters)
        self.activation = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = []
        for i in range(1, num_conv_layers + 1):
            layers.append(nn.Conv2d(in_channels=num_starting_filters, out_channels=num_starting_filters, kernel_size=3, stride=1, padding=1))
        self.conv_layers = nn.Sequential(*layers)

        layers = []
        # create number of blocks according to the num_blocks input
        for i in range(num_linear_layers):
            curr_in_channels = num_starting_filters // (2 ** i)
            curr_out_channels = num_starting_filters // (2 ** (i + 1))
            layers.append(nn.Linear(curr_in_channels, curr_out_channels))

            # every other layer is a batch norm layer
            if i % 2 == 0:
                layers.append(nn.BatchNorm1d(curr_out_channels))
                layers.append(nn.LeakyReLU(inplace=True))
                layers.append(nn.Dropout(p=dropout))
        
        self.linear_layers = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(num_starting_filters // (2 ** (num_linear_layers)), num_classes)

        self.temp = nn.Linear(5, 2)

        self.lr = lr
        self.track_wandb = track_wandb
        self.train_step_losses = []
        self.validation_step_losses = []
        self.train_step_acc = []
        self.validation_step_acc = []
        self.last_train_acc = 0
        self.last_train_loss = 0
    
    def forward(self, x):
        bob = self.prep(x)
        bob = self.norm_1(bob)
        bob = self.conv_layers(bob)
        
        # Initialize a list to store the pooled outputs for each filter
        pooled_outputs = []

        # Apply global max pooling for each filter
        for i in range(bob.shape[1]):  # Iterate over the channels (filters)
            single_channel_feature_map = bob[:, i, :, :]  # Get a single channel (filter)
            pooled_output = F.adaptive_max_pool2d(single_channel_feature_map, (1, 1))
            pooled_outputs.append(pooled_output)

        # Concatenate the pooled outputs along the channel dimension to get the final output
        pooled_outputs = torch.cat(pooled_outputs, dim=1)

        # The pooled_outputs tensor now contains the global max-pooled output for each filter
        pooled_output = pooled_outputs.squeeze(2).to("mps")


        bob = self.linear_layers(pooled_output)
        x = self.fc(bob)

        # # Create a sample input tensor (batch_size, channels, height, width)
        # input_data = torch.randn(1, 3, 32, 32)  # Example input with 3 channels

        # # Create a Conv2d layer with multiple filters (e.g., 5 filters)
        # conv_layer = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, stride=1, padding=1).to("mps")

        # # Apply the Conv2d layer to the input data
        # feature_maps = conv_layer(x)

        # # Initialize a list to store the pooled outputs for each filter
        # pooled_outputs = []

        # # Apply global max pooling for each filter
        # for i in range(feature_maps.shape[1]):  # Iterate over the channels (filters)
        #     single_channel_feature_map = feature_maps[:, i, :, :]  # Get a single channel (filter)
        #     pooled_output = F.adaptive_max_pool2d(single_channel_feature_map, (1, 1))
        #     pooled_outputs.append(pooled_output)

        # # Concatenate the pooled outputs along the channel dimension to get the final output
        # pooled_outputs = torch.cat(pooled_outputs, dim=1)

        # # The pooled_outputs tensor now contains the global max-pooled output for each filter
        # pooled_output = pooled_outputs.squeeze(2).to("mps")

        # x = self.temp(pooled_output)
        return x
    
    def training_step(self, batch, batch_idx):
        images, labels = batch

        # forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        self.train_step_losses.append(loss)
        
        # log accuracy
        _,preds = torch.max(outputs.data, 1)
        acc = (preds == labels).sum().item()
        acc /= outputs.size(dim=0)
        acc *= 100
        self.train_step_acc.append(acc)

        return {'loss':loss}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)
    
    def train_dataloader(self):
        train_loader, val_loader = dataloader.get_data()
        return train_loader
    
    def on_train_epoch_end(self):
        all_preds = self.train_step_losses
        avg_loss = sum(all_preds) / len(all_preds)
        
        all_acc = self.train_step_acc
        avg_acc = sum(all_acc) / len(all_acc)
        avg_acc = round(avg_acc, 2)

        self.last_train_acc = avg_acc
        self.last_train_loss = avg_loss

        # clear memory
        self.train_step_acc.clear()
        self.train_step_losses.clear()
        
        return {'train_loss':avg_loss}
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        # images = images.reshape(-1, 28 * 28)
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        self.validation_step_losses.append(loss.item())

        # log accuracy
        _,preds = torch.max(outputs.data, 1)
        acc = (preds == labels).sum().item()
        acc /= outputs.size(dim=0)
        acc *= 100
        self.validation_step_acc.append(acc)

        return {'val_loss':loss}
    
    def val_dataloader(self):
        train_loader, val_loader = dataloader.get_data()
        return val_loader
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.l1(x)
        x_hat = self.l2(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)
    
    def on_validation_epoch_end(self):
        all_preds = self.validation_step_losses
        all_acc = self.validation_step_acc

        avg_loss = sum(all_preds) / len(all_preds)
        avg_acc = sum(all_acc) / len(all_acc)
        avg_acc = round(avg_acc, 2)

        if self.track_wandb:
            wandb.log({"training_loss":self.last_train_loss,
                        "training_acc":self.last_train_acc,
                        "validation_loss":avg_loss,
                        "validation_acc":avg_acc})

        # clear memory
        self.validation_step_losses.clear()
        self.validation_step_acc.clear()

        return {'val_loss':avg_loss}

epochs = 5
train_loader, valid_loader = dataloader.get_data()

if __name__ == "__main__":
    model = food_classifier(lr=0.001, dropout=0.2, num_classes=2, track_wandb=False)

    trainer = Trainer(max_epochs = epochs, fast_dev_run=False)
    trainer.fit(model, train_loader, valid_loader)

# # Create a sample input tensor (batch_size, channels, height, width)
# input_data = torch.randn(1, 3, 32, 32)  # Example input with 3 channels

# # Create a Conv2d layer with multiple filters (e.g., 5 filters)
# conv_layer = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, stride=1, padding=1)

# # Apply the Conv2d layer to the input data
# feature_maps = conv_layer(input_data)

# # Initialize a list to store the pooled outputs for each filter
# pooled_outputs = []

# # Apply global max pooling for each filter
# for i in range(feature_maps.shape[1]):  # Iterate over the channels (filters)
#     single_channel_feature_map = feature_maps[:, i, :, :]  # Get a single channel (filter)
#     pooled_output = F.adaptive_max_pool2d(single_channel_feature_map, (1, 1))
#     pooled_outputs.append(pooled_output)

# # Concatenate the pooled outputs along the channel dimension to get the final output
# pooled_outputs = torch.cat(pooled_outputs, dim=1)

# # The pooled_outputs tensor now contains the global max-pooled output for each filter
# print(pooled_outputs.squeeze(2).shape)  # The shape will be (batch_size, num_filters, 1, 1)