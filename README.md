# Projet-DL-3A-CS

- [Projet-DL-3A-CS](#projet-dl-3a-cs)
  - [Repository organisation](#repository-organisation)
  - [Training codes](#training-codes)
    - [Models](#models)
    - [Utilitary](#utilitary)
  - [Repository usage](#repository-usage)
    - [Data](#data)
    - [Training](#training)
    - [Output](#output)


## Repository organisation

- `checkpoint/` : trained model weights, hyperparameters and learning data
- `vizualisation_img/` : gifs with model output and gradients
- `Task01_BrainTumour/` : folder with the dataset 

## Training codes  

### Models 

- `baseline_model.py` : baseline model implementation
- `baseline_model_aug.py` : baseline model with data augmentation
- `baseline_model_focal.py` : baseline model with focal loss instead of MSE loss
- `baseline_model_skip_connections.py` : baseline model with simple skip connection
- `baseline_model_skip_connections.py` : baseline model with simple skip connection
  
### Utilitary 
- `utils.py` : utilitary functions, common for most models
- `metrics.py` : custom loss implementation, common for most models
- `utils_gan.py` : utilitary functions for the GAN model
- `metrics_gan.py` : metrics for the GAN model
- `baseline_model_skip_connections.py` : baseline model with simple skip connection
- `load_log` : sample code to open pickle file from training 
  

## Repository usage 

### Data

Download the data from the official website (BRATS 2016, 2017), or from this [link](https://centralesupelec-my.sharepoint.com/:f:/g/personal/tristan_beolet_student-cs_fr/ErkVSEpIkuxGow294KqM9AEBNak8PfJNZ3Q-Ffq0yPK_1g?e=4huX1h) (might faster). The folder should be at the root of this repository and called : `Task01_BrainTumour`

### Training

To train a model, just run the following command 
```bash 
python <file.py>
```


### Output

All the training data will be saved in the `checkpoints/` folder. You can identify different runs with the timestamp of when the training started. 

Figures with loss over epochs will be saved for both training and testing with the desired loss. 

### Visualization

To visualize results from a model, you can use ```python <data_viz.py>``` command. It will produce animated GIFs, displaying the MR slices, ground truth according to doctors, and the prediction from the selected model. You can also choose to visualize the gradient informations. 

Several variables can be changed in this file : 
- `MODEL` : fill in path to saved model to load it and use it. Warning : if you are using 'nvidia vnet' model, you need to change the import line 2, to load the right Vnet class.
- `MODEL_NAME` : model name, to be displayed on images, and used when saving.
- `SEED` : use 0 to get the same train/test split as in models
- `WIDTH, HEIGHT, DEPTH` : used to transform images
- `TRAINING_SPLIT_RATIO` : Choose it as in model training to get same results
- `NB_DISPLAYED` : Number of GIFs to be displayed, for both train and test dataset
- `SAVE_GIFS` : If `True`, images will be saved in `visualization_img` folder
- `DISPLAY_GRADIENT` : If True, an image will be added to the GIF to display gradient
- `TARGET_LAYER` : layers to use for gradient cam

  
