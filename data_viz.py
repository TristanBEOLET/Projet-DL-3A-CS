import logging
from baseline_model import VNet

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
from torch import nn

import data_utils
import matplotlib.pyplot as plt

from torchcam.methods import GradCAMpp
import time as t

import numpy as np
import utils



def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info("Loading data")

    ###Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = torch.load('./checkpoints/baseline_224x224/20230327-151415_best_model.pt', map_location=torch.device(device)).to(device)
    MODEL.eval()
    
    SEED = 0
    BATCH_SIZE = 1
    WIDTH = 224
    HEIGHT = 224
    DEPTH = 20
    TRAINING_SPLIT_RATIO = 0.8
    NB_DISPLAYED = 1
    SAVE_GIFS = True
    DISPLAY_GRADIENT = False
    TARGET_LAYER = [MODEL.deconv4, MODEL.deconv5, MODEL.deconv6]

    torch.manual_seed(SEED)
    # Data
    transform = transforms.Compose(
        [
            data_utils.Resize3D((WIDTH, HEIGHT, DEPTH)),
        ]
    )

    dataset = data_utils.MSD_Brain_Tumor("Task01_BrainTumour", transform=transform)

    nb_train = int(TRAINING_SPLIT_RATIO * len(dataset))
    train, test = random_split(dataset, [nb_train, len(dataset) - nb_train])

    train_dataloader = DataLoader(
        train, batch_size=BATCH_SIZE, shuffle=True
    )
    test_dataloader = DataLoader(
        test, batch_size=BATCH_SIZE, shuffle=False
    )

    if DISPLAY_GRADIENT:
        cam_extractor = GradCAMpp(MODEL, input_shape=(WIDTH, HEIGHT, DEPTH), target_layer = TARGET_LAYER) 
    
    for j, dataloader in enumerate([train_dataloader, test_dataloader]):
        dataloader_name = 'train' if j == 0 else 'test'
        for i, data in enumerate(dataloader):
            if i >= NB_DISPLAYED:
                break
            X = data[0].float()[:, None, :, :, :].to(device)
            y = data[1].float()[:, None, :, :, :].to(device)
            y_pred = MODEL(X)


            if DISPLAY_GRADIENT:
                dice_loss, mse_loss = utils.get_scores(y, y_pred)
                loss = dice_loss + mse_loss
                scores = loss.unsqueeze(0).unsqueeze(0)
                all_cams = cam_extractor(0, scores)
                cams = cam_extractor.fuse_cams(all_cams)
                cam_resized = nn.functional.interpolate(cams[0, :, :, :].unsqueeze(0).unsqueeze(0), size=X[0, 0,:,:, :].shape, mode='trilinear')

            y_pred = y_pred[0][0].permute(2,0,1).detach()
            X = X[0][0].permute(2,0,1)
            y = y[0][0].permute(2,0,1)
            
            utils.generate_gif(X, y, y_pred, image_name = f'{dataloader_name}_n°{i}', model_name='best_224_skip_20230327', save_gifs=SAVE_GIFS, X_grad_cam = cam_resized if DISPLAY_GRADIENT else None)
if __name__ == "__main__":
    main()