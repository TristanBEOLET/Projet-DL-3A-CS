import matplotlib.pyplot as plt
import numpy as np
import torch


def dice(inputs: torch.tensor, targets: torch.tensor) -> float:
    """Compute dice score between two tensors.

    Args:
        inputs (torch.tensor): prediction
        targets (torch.tensor): target

    Returns:
        float: dice score
    """
    smooth = 1.0
    inputs = inputs.contiguous().view(inputs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)

    intersection = (inputs * targets).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        inputs.sum(dim=1) + targets.sum(dim=1) + smooth
    )
    return dice.mean()


def plot_training_metrics(time_stamp):
    """Plot and saves training losses and scores."""
    training_data = np.load(f"checkpoints/{time_stamp}_training_run.npz")
    plt.plot(training_data["train_dice_scores"], label="Train Dice score")
    plt.plot(training_data["test_dice_scores"], label="Test Dice score")
    plt.legend()
    plt.title("Dice score during training")
    plt.savefig(f"checkpoints/{time_stamp}_dice.svg")
    plt.show()
    plt.semilogy(training_data["train_mse_scores"], label="Train MSE loss")
    plt.semilogy(training_data["test_mse_scores"], label="Test MSE loss")
    plt.legend()
    plt.title("MSE loss during training")
    plt.savefig(f"checkpoints/{time_stamp}_mse.svg")
    plt.show()


if __name__ == "__main__":
    plot_training_metrics("20230321-004136")
