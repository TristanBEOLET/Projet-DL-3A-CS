import glob
import os

import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class Resize3D(object):
    """Resize 3D images."""
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        resample = F.interpolate(sample[None,:,:,:].unsqueeze(0), size=self.size, mode='trilinear', align_corners=False)
        return resample.squeeze(0)[0,:,:,:]



class MSD_Brain_Tumor(Dataset):
    """Pytorch dataset for brain tumor dataset."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = list(glob.glob(f"{root_dir}/imagesTr/*.nii.gz"))#os.listdir(os.path.join(root_dir, "imagesTr"))
        self.label_files = list(glob.glob(f"{root_dir}/labelsTr/*.nii.gz"))#os.listdir(os.path.join(root_dir, "labelsTr"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = self.label_files[idx]
        image = nib.load(image_path)
        image_data = image.get_fdata()[:,:,:,0]
        image_data = torch.tensor(image_data)
        label = nib.load(label_path)
        label_data = label.get_fdata()
        label_data = torch.tensor(label_data)
        if self.transform:
            image_tensor = self.transform(image_data)
            label_tensor = self.transform(label_data)
            return (image_tensor, label_tensor)
        return (image_data, label_data)