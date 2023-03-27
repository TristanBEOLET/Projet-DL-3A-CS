import logging
import pickle
import time

import data_utils
import matplotlib.pyplot as plt
import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms

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
        batch_size=64,
        training_split_ratio=0.8,
        lr=0.001,
        weight_decay=1e-5,
        treshold=0.5,
        image_size=(128, 128, 20),
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
            data_utils.Resize3D(hyperparameters.image_size),
        ]
    )

    dataset = data_utils.MSD_Brain_Tumor("Task01_BrainTumour", transform=transform)
    nb_train = int(hyperparameters.training_split_ratio * len(dataset))
    train, test = random_split(dataset, [nb_train, len(dataset) - nb_train])

    train_dataloader = DataLoader(
        train, batch_size=hyperparameters.batch_size, shuffle=True, num_workers=15
    )
    test_dataloader = DataLoader(
        test, batch_size=hyperparameters.batch_size, shuffle=False, num_workers=15
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
