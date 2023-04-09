import logging
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms

import data_utils
import metrics
import utils

torch.manual_seed(0)


class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()
        # Convolution layers
        self.batch_norm0 = nn.BatchNorm3d(1)
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=2, stride=2)
        # self.conv5 = nn.Conv3d(128, 256, kernel_size=2, stride=1)

        # Deconvolution layers
        # self.deconv5 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=1)
        self.deconv4 = nn.ConvTranspose3d(
            128, 64, kernel_size=2, stride=2, output_padding=(1, 1, 0)
        )
        # self.deconv4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.deconv1 = nn.ConvTranspose3d(16, 1, kernel_size=5, stride=1)

        # Prelu layers
        self.down_prelu1 = nn.PReLU()
        self.down_prelu2 = nn.PReLU()
        self.down_prelu3 = nn.PReLU()
        self.down_prelu4 = nn.PReLU()
        self.down_prelu5 = nn.PReLU()

        self.up_prelu1 = nn.PReLU()
        self.up_prelu2 = nn.PReLU()
        self.up_prelu3 = nn.PReLU()
        self.up_prelu4 = nn.PReLU()
        self.up_prelu5 = nn.PReLU()

        # Convolve
        self.horizontal_conv1 = nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1)
        self.horizontal_conv2 = nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1)
        self.horizontal_conv3 = nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Define the forward pass of the V-Net model
        x = self.batch_norm0(x)
        x = self.conv1(x)
        x16 = self.down_prelu1(x)
        x = self.conv2(x16)
        x32 = self.down_prelu2(x)
        x = self.conv3(x32)
        x64 = self.down_prelu3(x)
        x = self.conv4(x64)
        x128 = self.down_prelu4(x)
        # x = self.conv5(x128)
        # x256 = self.down_prelu5(x)

        # x = self.deconv5(x256)
        # x = self.up_prelu1(x)
        x = self.deconv4(x128)
        x = self.up_prelu2(x)
        x = self.horizontal_conv1(torch.cat([x, x64], 1))
        x = self.deconv3(x)
        x = self.up_prelu3(x)
        x = self.horizontal_conv2(torch.cat([x, x32], 1))
        x = self.deconv2(x)
        x = self.up_prelu4(x)
        x = self.horizontal_conv3(torch.cat([x, x16], 1))
        x = self.deconv1(x)

        return x


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info("Loading data")

    # Saving start time to use as a timestamp
    time_stamp = time.strftime("%Y%m%d-%H%M%S")

    # Constants
    hyperparameters = utils.Hyperparameters(
        num_epochs=500,
        batch_size=32,
        training_split_ratio=0.8,
        lr=0.001,
        weight_decay=1e-5,
        treshold=0.5,
    )
    hyperparameters.save(time_stamp=time_stamp)

    logging.debug(f"Learning rate : {hyperparameters.lr}")
    logging.debug(f"weight decay : {hyperparameters.weight_decay}")
    logging.debug(f"Prediction threshold : {hyperparameters.treshold}")
    logging.debug(f"Batch size : {hyperparameters.batch_size}")
    logging.debug(f"Training split ratio : {hyperparameters.training_split_ratio}")

    # Data
    transform = transforms.Compose(
        [
            data_utils.Resize3D((128, 128, 20)),
        ]
    )

    dataset = data_utils.MSD_Brain_Tumor("Task01_BrainTumour", transform=transform)
    nb_train = int(hyperparameters.training_split_ratio * len(dataset))
    train, test = random_split(dataset, [nb_train, len(dataset) - nb_train])

    train_dataloader = DataLoader(
        train, batch_size=hyperparameters.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test, batch_size=hyperparameters.batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VNet().float().to(device)

    (
        train_dice_scores,
        train_mse_scores,
        test_dice_scores,
        test_mse_scores,
    ) = utils.train_model(
        model,
        hyperparameters=hyperparameters,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        time_stamp=time_stamp,
    )

    metrics.plot_training_metrics(time_stamp)


if __name__ == "__main__":
    main()
