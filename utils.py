import logging
import pickle
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from matplotlib import animation
from matplotlib import pyplot as plt

import metrics
import utils


class Hyperparameters:
    def __init__(
        self,
        **kwargs,  # num_epochs, batch_size, training_split_ratio, lr, weight_decay, treshold
    ) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        """self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.training_split_ratio = training_split_ratio
        self.lr = lr
        self.weight_decay = weight_decay
        self.treshold = treshold"""

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


def test_model_focal(
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
            mse_loss = criterion(outputs, labels, reduction="mean")
            loss = dice_loss + mse_loss
            test_dice_loss_sum += dice_loss.item()
            test_mse_loss_sum += mse_loss.item()

    return 1 - (test_dice_loss_sum / len(test_dataloader)), test_mse_loss_sum / len(
        test_dataloader
    )


def get_scores(y, y_pred, threshold=0.5):
    threshold = 0.5
    criterion = nn.MSELoss(reduction="mean")
    dice_loss = 1 - metrics.dice(y_pred > threshold, y)
    mse_loss = criterion(y_pred, y)

    return dice_loss, mse_loss


def train_model(
    model, hyperparameters, train_dataloader, test_dataloader, device, time_stamp
):
    """Train model, saves the model with the best validation dice score, and final model."""
    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(
        model.parameters(),
        lr=hyperparameters.lr,
        weight_decay=hyperparameters.weight_decay,
    )

    # Keeping the losses in memory to plot training evolution
    train_dice_losses = []
    train_mse_losses = []

    train_dice_scores = []
    train_mse_scores = []

    test_dice_scores = []
    test_mse_scores = []

    logging.info(f"Starting training for {hyperparameters.num_epochs} epochs")
    for epoch in range(hyperparameters.num_epochs):
        model.train()
        dice_loss_sum = 0
        mse_loss_sum = 0
        for i, data in enumerate(train_dataloader):
            inputs = data[0].float()[:, None, :, :, :].to(device)
            labels = (data[1] != 0.0).float()[:, None, :, :, :].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            dice_loss = 1 - metrics.dice(outputs > hyperparameters.treshold, labels)
            mse_loss = criterion(outputs, labels)
            loss = dice_loss + mse_loss
            loss.backward()
            dice_loss_sum += dice_loss.item()
            mse_loss_sum += mse_loss.item()
            optimizer.step()
        train_dice_losses.append(dice_loss_sum)
        train_mse_losses.append(mse_loss_sum)

        dice_score = 1 - (dice_loss_sum / len(train_dataloader))
        mse_score = mse_loss_sum / len(train_dataloader)
        train_dice_scores.append(dice_score)
        train_mse_scores.append(mse_score)

        test_dice_score, test_mse_score = utils.test_model(
            model=model,
            test_dataloader=test_dataloader,
            criterion=criterion,
            treshold=hyperparameters.treshold,
            device=device,
        )

        logging.info(
            f"Epoch {epoch+1}/{hyperparameters.num_epochs}, Train dice: {dice_score:.4f}, Train MSE : {mse_score:.5f}, Test dice: {test_dice_score:.4f}, Test MSE : {test_mse_score:.5f}"
        )

        if test_dice_score > max(test_dice_scores, default=0):
            logging.info(f"Saving model for test dice score : {test_dice_score:.4f}")
            torch.save(model, f"checkpoints/{time_stamp}_best_model.pt")

        test_dice_scores.append(test_dice_score)
        test_mse_scores.append(test_mse_score)

    torch.save(model, f"checkpoints/{time_stamp}_final_model.pt")
    save_training_data(
        train_dice_scores=train_dice_scores,
        train_mse_scores=train_mse_scores,
        test_dice_scores=test_dice_scores,
        test_mse_scores=test_mse_scores,
        time_stamp=time_stamp,
    )
    return train_dice_scores, train_mse_scores, test_dice_scores, test_mse_scores


def train_model_focal(
    model, hyperparameters, train_dataloader, test_dataloader, device, time_stamp
):
    """Train model, saves the model with the best validation dice score, and final model."""
    criterion = torchvision.ops.sigmoid_focal_loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=hyperparameters.lr,
        weight_decay=hyperparameters.weight_decay,
    )

    # Keeping the losses in memory to plot training evolution
    train_dice_losses = []
    train_mse_losses = []

    train_dice_scores = []
    train_mse_scores = []

    test_dice_scores = []
    test_mse_scores = []

    logging.info(f"Starting training for {hyperparameters.num_epochs} epochs")
    for epoch in range(hyperparameters.num_epochs):
        model.train()
        dice_loss_sum = 0
        mse_loss_sum = 0
        for i, data in enumerate(train_dataloader):
            inputs = data[0].float()[:, None, :, :, :].to(device)
            labels = (data[1] != 0.0).float()[:, None, :, :, :].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            dice_loss = 1 - metrics.dice(outputs > hyperparameters.treshold, labels)
            mse_loss = criterion(outputs, labels, reduction="mean")
            loss = dice_loss + mse_loss
            loss.backward()
            dice_loss_sum += dice_loss.item()
            mse_loss_sum += mse_loss.item()
            optimizer.step()
        train_dice_losses.append(dice_loss_sum)
        train_mse_losses.append(mse_loss_sum)

        dice_score = 1 - (dice_loss_sum / len(train_dataloader))
        mse_score = mse_loss_sum / len(train_dataloader)
        train_dice_scores.append(dice_score)
        train_mse_scores.append(mse_score)

        test_dice_score, test_mse_score = utils.test_model_focal(
            model=model,
            test_dataloader=test_dataloader,
            criterion=criterion,
            treshold=hyperparameters.treshold,
            device=device,
        )

        logging.info(
            f"Epoch {epoch+1}/{hyperparameters.num_epochs}, Train dice: {dice_score:.4f}, Train Focal : {mse_score:.5f}, Test dice: {test_dice_score:.4f}, Test Focal : {test_mse_score:.5f}"
        )

        if test_dice_score > max(test_dice_scores, default=0):
            logging.info(f"Saving model for test dice score : {test_dice_score:.4f}")
            torch.save(model, f"checkpoints/{time_stamp}_best_model.pt")

        test_dice_scores.append(test_dice_score)
        test_mse_scores.append(test_mse_score)

    torch.save(model, f"checkpoints/{time_stamp}_final_model.pt")
    save_training_data(
        train_dice_scores=train_dice_scores,
        train_mse_scores=train_mse_scores,
        test_dice_scores=test_dice_scores,
        test_mse_scores=test_mse_scores,
        time_stamp=time_stamp,
    )
    return train_dice_scores, train_mse_scores, test_dice_scores, test_mse_scores


def generate_gif(
    X,
    y,
    y_pred,
    threshold=0.5,
    image_name="train",
    model_name="",
    X_grad_cam=None,
    save_gifs=True,
):
    ncols = 3 if isinstance(X_grad_cam, type(None)) else 4
    fig, axs = plt.subplots(ncols=ncols, sharey=True, figsize=(6, 2))
    image = axs[0].imshow(
        X[0, :, :], animated=True, cmap="gray", vmin=X.min(), vmax=X.max()
    )

    # dupliquer sur 3 canaux + supprimer l'endroit où on a un masque sur les 3 canaux, et le faire apparaître en vert
    X_repeated = X.unsqueeze(1).repeat(1, 3, 1, 1).permute(0, 2, 3, 1) / X.max()

    X_segmentation_truth = torch.where(
        y.unsqueeze(3).repeat(1, 1, 1, 3) == 0, X_repeated.float(), 0
    )
    X_segmentation_truth[:, :, :, 1] = torch.where(
        y == 0, X_segmentation_truth[:, :, :, 1], y.float() / 3
    )
    # Peut-être changer pour mettre vert/orange/rouge en fonction de la valeur ?
    X_segmentation_predicted = torch.where(
        y_pred.unsqueeze(3).repeat(1, 1, 1, 3) <= threshold, X_repeated.float(), 0
    )
    X_segmentation_predicted[:, :, :, 0] = torch.where(
        y_pred <= threshold, X_segmentation_predicted[:, :, :, 1], y_pred / y_pred.max()
    )
    # idem
    if not isinstance(X_grad_cam, type(None)):
        masked_cam = np.ma.masked_where(X_grad_cam <= 0.1, X_grad_cam)[0, 0, :, :, :]

    segmentation_truth = axs[1].imshow(
        X_segmentation_truth[0, :, :, :],
        animated=True,
        cmap="Greens",
        vmin=X_segmentation_truth.min(),
        vmax=X_segmentation_truth.max(),
    )

    segmentation_predicted = axs[2].imshow(
        X_segmentation_predicted[0, :, :, :],
        animated=True,
        cmap="gray",
        vmin=X_segmentation_predicted.min(),
        vmax=X_segmentation_predicted.max(),
    )

    if not isinstance(X_grad_cam, type(None)):
        base_grad_cam = axs[3].imshow(
            X[0, :, :], animated=True, cmap="gray", vmin=X.min(), vmax=X.max()
        )
        mask_grad_cam = axs[3].imshow(
            masked_cam[:, :, 0], alpha=0.6, vmin=masked_cam.min(), vmax=masked_cam.max()
        )

    def init_function():
        image.set_data(X[0, :, :])
        segmentation_truth.set_data(X_segmentation_truth[0, :, :, :])
        segmentation_predicted.set_data(X_segmentation_predicted[0, :, :, :])
        if not isinstance(X_grad_cam, type(None)):
            base_grad_cam.set_data(X[0, :, :])
            mask_grad_cam.set_data(masked_cam[:, :, 0])
            return (
                image,
                segmentation_truth,
                segmentation_predicted,
                base_grad_cam,
                mask_grad_cam,
            )
        return (
            image,
            segmentation_truth,
            segmentation_predicted,
        )

    def animate(i):
        image.set_array(X[i, :, :])
        segmentation_truth.set_array(X_segmentation_truth[i, :, :, :])
        segmentation_predicted.set_array(X_segmentation_predicted[i, :, :, :])
        if not isinstance(X_grad_cam, type(None)):
            base_grad_cam.set_array(X[i, :, :])
            mask_grad_cam.set_array(masked_cam[:, :, i])
            return (
                image,
                segmentation_truth,
                segmentation_predicted,
                base_grad_cam,
                mask_grad_cam,
            )
        return (
            image,
            segmentation_truth,
            segmentation_predicted,
        )

    # calling animation function of matplotlib
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init_function,
        frames=np.shape(X)[0],  # amount of frames being animated
        interval=200,  # update every second
        blit=True,
    )
    fig.suptitle(
        f"Patient : {image_name}, threshold = {threshold}, model={model_name}",
        fontsize=11,
    )
    if save_gifs:
        anim.save(
            f"./vizualisation_img/img_{model_name}_{image_name}.gif", writer="Pillow"
        )  # save as gif
    plt.show()
