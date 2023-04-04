import logging
from baseline_model import VNet

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from torchcam.methods import SmoothGradCAMpp, GradCAMpp

import metrics
import data_utils
import numpy as np

import matplotlib.pyplot as plt

torch.manual_seed(0)

def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info("Loading data")

    # Data
    transform = transforms.Compose(
        [
            data_utils.Resize3D((128, 128, 20)),
        ]
    )
    BATCH_SIZE = 4
    dataset = data_utils.MSD_Brain_Tumor("Task01_BrainTumour", transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load('./checkpoints/skip_connections/20230321-230008_best_model.pt', map_location=torch.device(device)).to(device)
    print(model)
    target_layer = [model.conv3, model.deconv4, model.deconv5, model.deconv6]
    cam_extractor = GradCAMpp(model, input_shape=(128,128,20), target_layer = target_layer) 
    for i, data in enumerate(dataloader):
        X = data[0].float()[:, None, :, :, :].to(device)
        y = (data[1] != 0.0).float()[:, None, :, :, :].to(device)
        # Preprocess your data and feed it to the model
        y_pred = model(X)
        threshold = 0.5
        criterion = nn.MSELoss(reduction="mean")
        # y_pred = (y_pred>=threshold)*1
        dice_loss = 1 - metrics.dice(y_pred > threshold, y)
        mse_loss = criterion(y_pred, y)
        loss = dice_loss + mse_loss
        
        scores = loss.unsqueeze(0).unsqueeze(0)

        # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(0, scores)
        cams = cam_extractor.fuse_cams(activation_map)
        for name, cam in zip(cam_extractor.target_names, activation_map):
            for i in range(1):
                cam_resized = nn.functional.interpolate(cams[i, :, :, :].unsqueeze(0).unsqueeze(0), size=X[i, 0,:,:, :].shape, mode='trilinear')
                z = int(X.shape[-1]/2)
                X_repeated = X[i, 0,:,:, z].repeat(3, 1, 1)/X[i, 0,:,:, z].max()
                masked_cam = np.ma.masked_where(cam_resized <= 0.1, cam_resized)[0, 0, :, :, :]

                # Overlay the two images
                fig, ax = plt.subplots()
                ax.imshow(X[:, :, z], cmap='gray')
                ax.imshow(masked_cam[:, :, z], alpha=0.6)
                plt.show()
                # result = overlay_mask(to_pil_image(X_repeated), to_pil_image(cam_resized[0, 0, :, :, z], mode='F'), alpha=0.4)
                # plt.imshow(result); plt.axis('off'); plt.title(name); plt.show()
            break
        break


if __name__ == "__main__":
    main()