import numpy as np

data = np.load("checkpoints/baseline/20230321-004136_training_run.npz")

# Listing all the files stored in the npz file
print("Available files : ", data.__dict__["files"])

# Example for one of the files
print(data["train_dice_scores"])
