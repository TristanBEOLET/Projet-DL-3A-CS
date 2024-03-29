import logging
import pickle
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import metrics
import utils


class Hyperparameters:
    def __init__(
        self, num_epochs, batch_size, training_split_ratio, lr, weight_decay, treshold
    ) -> None:
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.training_split_ratio = training_split_ratio
        self.lr = lr
        self.weight_decay = weight_decay
        self.treshold = treshold

    def save(self, time_stamp):
        pickle.dump(
            self,
            open(
                f"checkpoints/{time_stamp}_hyperpameters.pkl",
                "wb",
            ),
        )


def save_training_data(
    train_dice_scores: list,
    train_mse_scores: list,
    train_discri_scores: list,
    test_dice_scores: list,
    test_mse_scores: list,
    time_stamp: list,
) -> None:
    """Save all training losses and scores to checkpoint folder.

    Args:
        train_dice_scores (list): _description_
        train_mse_scores (list): _description_
        test_dice_scores (list): _description_
        test_mse_scores (list): _description_
        time_stamp (list): _description_
    """
    np.savez(
        f"checkpoints/{time_stamp}_training_run.npz",
        train_dice_scores=train_dice_scores,
        train_mse_scores=train_mse_scores,
        train_discri_scores=train_discri_scores,
        test_dice_scores=test_dice_scores,
        test_mse_scores=test_mse_scores,
    )


def test_model(
    model, test_dataloader, criterion, treshold, device
) -> Tuple[float, float]:
    """Test model on testing dataloader."""
    model.eval()
    test_dice_loss_sum = 0
    test_mse_loss_sum = 0

    with torch.no_grad():
        for data in test_dataloader:
            inputs = data[0].float()[:, None, :, :, :].to(device)
            labels = (data[1] != 0.0).float()[:, None, :, :, :].to(device)
            outputs = model(inputs)
            dice_loss = 1 - metrics.dice(outputs > treshold, labels)
            mse_loss = criterion(outputs, labels)
            loss = dice_loss + mse_loss
            test_dice_loss_sum += dice_loss.item()
            test_mse_loss_sum += mse_loss.item()

    return 1 - (test_dice_loss_sum / len(test_dataloader)), test_mse_loss_sum / len(
        test_dataloader
    )


def train_model(
    generator,
    discriminator,
    hyperparameters_gn,
    hyperparameters_discri,
    train_dataloader,
    test_dataloader,
    device,
    time_stamp,
):
    """Train model, saves the model with the best validation dice score, and final model."""
    criterion_gn = nn.MSELoss(reduction="mean")
    criterion_dc = nn.BCELoss()
    optimizer_gn = optim.Adam(
        generator.parameters(),
        lr=hyperparameters_gn.lr,
        weight_decay=hyperparameters_gn.weight_decay,
    )

    optimizer_discri = optim.Adam(
        generator.parameters(),
        lr=hyperparameters_discri.lr,
        weight_decay=hyperparameters_discri.weight_decay,
    )

    # Keeping the losses in memory to plot training evolution
    train_dice_losses = []
    train_mse_losses = []

    train_dice_scores = []
    train_mse_scores = []

    test_dice_scores = []
    test_mse_scores = []

    train_discri_losses = []
    train_discri_scores = []

    logging.info(f"Starting training for {hyperparameters_gn.num_epochs} epochs")
    for epoch in range(hyperparameters_gn.num_epochs):
        generator.train()
        discriminator.train()
        dice_loss_sum = 0
        mse_loss_sum = 0
        discri_loss_sum = 0
        for i, data in enumerate(train_dataloader):
            inputs = data[0].float()[:, None, :, :, :].to(device)
            labels = (data[1] != 0.0).float()[:, None, :, :, :].to(device)
            optimizer_gn.zero_grad()
            optimizer_discri.zero_grad()
            outputs = generator(inputs)
            dice_loss = 1 - metrics.dice(outputs > hyperparameters_gn.treshold, labels)
            mse_loss = criterion_gn(outputs, labels)
            loss_gene = dice_loss + mse_loss
            loss_gene.backward()
            discri_real_loss = criterion_dc(
                discriminator(labels), torch.ones(labels.size(0), 1).to(device)
            )
            discri_fake_loss = criterion_dc(
                discriminator(outputs.detach()),
                torch.zeros(outputs.size(0), 1).to(device),
            )
            loss_discri = discri_real_loss + discri_fake_loss
            loss_discri.backward()
            optimizer_discri.step()
            dice_loss_sum += dice_loss.item()
            mse_loss_sum += mse_loss.item()
            discri_loss_sum += loss_discri.item()
            optimizer_gn.step()
            optimizer_discri.step()
        train_dice_losses.append(dice_loss_sum)
        train_mse_losses.append(mse_loss_sum)
        train_discri_losses.append(discri_loss_sum)

        dice_score = 1 - (dice_loss_sum / len(train_dataloader))
        mse_score = mse_loss_sum / len(train_dataloader)
        discri_score = discri_loss_sum / len(train_dataloader)
        train_dice_scores.append(dice_score)
        train_mse_scores.append(mse_score)
        train_discri_scores.append(discri_score)

        test_dice_score, test_mse_score = utils.test_model(
            model=generator,
            test_dataloader=test_dataloader,
            criterion=criterion_gn,
            treshold=hyperparameters_gn.treshold,
            device=device,
        )

        logging.info(
            f"Epoch {epoch+1}/{hyperparameters_gn.num_epochs}, Train dice: {dice_score:.4f}, Train MSE : {mse_score:.5f}, Train Discri : {discri_score:.5f}, Test dice: {test_dice_score:.4f}, Test MSE : {test_mse_score:.5f}"
        )

        if test_dice_score > max(test_dice_scores, default=0):
            logging.info(f"Saving model for test dice score : {test_dice_score:.4f}")
            torch.save(generator, f"checkpoints/{time_stamp}_best_model.pt")

        test_dice_scores.append(test_dice_score)
        test_mse_scores.append(test_mse_score)

    torch.save(generator, f"checkpoints/{time_stamp}_final_model.pt")
    save_training_data(
        train_dice_scores=train_dice_scores,
        train_mse_scores=train_mse_scores,
        train_discri_scores=train_discri_scores,
        test_dice_scores=test_dice_scores,
        test_mse_scores=test_mse_scores,
        time_stamp=time_stamp,
    )
    return (
        train_dice_scores,
        train_mse_scores,
        train_discri_scores,
        test_dice_scores,
        test_mse_scores,
    )
