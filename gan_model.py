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
import metrics_gan
import utils_gan

torch.manual_seed(0)


class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()
        # Define the layers of the V-Net model
        self.batch_norm0 = nn.BatchNorm3d(1)
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1)
        self.deconv4 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=1)
        self.deconv5 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2)
        self.deconv6 = nn.ConvTranspose3d(
            32, 1, kernel_size=3, stride=2, output_padding=(1, 1, 1)
        )

    def forward(self, x):
        # Define the forward pass of the V-Net model
        x = self.batch_norm0(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.deconv4(x)
        x = F.relu(x)
        x = self.deconv5(x)
        x = F.relu(x)
        x = self.deconv6(x)
        return x


class Modeldiscriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Modeldiscriminator, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv3d(512, 1, kernel_size=1, stride=1, padding=0)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.linear = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.leakyrelu(x)
        x = self.conv3(x)
        x = self.leakyrelu(x)
        x = self.conv4(x)
        x = self.leakyrelu(x)
        x = self.conv5(x)
        x = self.linear(x.flatten(start_dim=1))
        x = self.sigmoid(x)
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
    hyperparameters_gn = utils_gan.Hyperparameters(
        num_epochs=500,
        batch_size=32,
        training_split_ratio=0.8,
        lr=0.001,
        weight_decay=1e-5,
        treshold=0.5,
    )
    hyperparameters_gn.save(time_stamp=time_stamp)

    hyperparameters_dc = utils_gan.Hyperparameters(
        num_epochs=500,
        batch_size=32,
        training_split_ratio=0.8,
        lr=0.001,
        weight_decay=1e-5,
        treshold=0.5,
    )
    hyperparameters_dc.save(time_stamp=time_stamp)

    logging.debug(f"Learning rate generator: {hyperparameters_gn.lr}")
    logging.debug(f"weight decay generator: {hyperparameters_gn.weight_decay}")
    logging.debug(f"Prediction threshold generator: {hyperparameters_gn.treshold}")
    logging.debug(f"Batch size generator: {hyperparameters_gn.batch_size}")
    logging.debug(
        f"Training split ratio generator : {hyperparameters_gn.training_split_ratio}"
    )

    logging.debug(f"Learning rate discriminator: {hyperparameters_dc.lr}")
    logging.debug(f"weight decay discriminator: {hyperparameters_dc.weight_decay}")
    logging.debug(f"Prediction threshold discriminator: {hyperparameters_dc.treshold}")
    logging.debug(f"Batch size discriminator: {hyperparameters_dc.batch_size}")
    logging.debug(
        f"Training split ratio discriminator : {hyperparameters_dc.training_split_ratio}"
    )

    # Data
    transform = transforms.Compose(
        [
            data_utils.Resize3D((128, 128, 20)),
        ]
    )

    dataset = data_utils.MSD_Brain_Tumor("Task01_BrainTumour", transform=transform)
    nb_train = int(hyperparameters_gn.training_split_ratio * len(dataset))
    train, test = random_split(dataset, [nb_train, len(dataset) - nb_train])

    train_dataloader = DataLoader(
        train, batch_size=hyperparameters_gn.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test, batch_size=hyperparameters_gn.batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = VNet().float().to(device)
    discriminator = Modeldiscriminator().float().to(device)

    (
        train_dice_scores,
        train_mse_scores,
        train_discri_scores,
        test_dice_scores,
        test_mse_scores,
    ) = utils_gan.train_model(
        generator,
        discriminator,
        hyperparameters_gn=hyperparameters_gn,
        hyperparameters_discri=hyperparameters_dc,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        time_stamp=time_stamp,
    )

    metrics_gan.plot_training_metrics(time_stamp)


if __name__ == "__main__":
    main()
