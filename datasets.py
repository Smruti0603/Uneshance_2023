import glob
import random
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)

#class ImageDataset(Dataset):
 #   def __init__(self, root, hr_shape,path = "/workspace/storage/HIGH_QUALITY/"):
  #      hr_height, hr_width = hr_shape
        #df=pd.read_csv(csv_file, header=None)
        # Transforms for low resolution images and high resolution images
    #    self.lr_transform = transforms.Compose(
     #       [
      #          transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
       #         transforms.ToTensor(),
        #        transforms.Normalize(mean, std),
         #   ]
       # )
      #  self.hr_transform = transforms.Compose(
       #     [
        #        transforms.Resize((hr_height, hr_height), Image.BICUBIC),
         #       transforms.ToTensor(),
          #      transforms.Normalize(mean, std),
           # ]
       # )
       # self.path = path
       # self.files = os.listdir(path)
        #self.file_names = list(df[0])
        #self.csv_file=df

   # def __getitem__(self, index):
    #    img_path = self.files[index]
     #   img = Image.open(os.path.join(self.path, img_path)).convert("RGB")
      #  img_lr = self.lr_transform(img)
      #  img_hr = self.hr_transform(img)
      #  return {"lr": img_lr, "hr": img_hr}

  #  def __len__(self):
   #     return len(self.files
#class ImageDataset(Dataset):
#    def __init__(self, root, hr_shape, path="/workspace/storage/HIGH_QUALITY/"):
#        hr_height, hr_width = hr_shape
        # Transforms for high resolution images only
#        self.hr_transform = transforms.Compose(
     #       [
 #               transforms.Resize((hr_height, hr_height), Image.BICUBIC),
  #              transforms.ToTensor(),
   #             transforms.Normalize(mean, std),
    #        ]
     #   )
      #  self.path = path
       # self.files = os.listdir(path)

#    def __getitem__(self, index):
 #       img_path = self.files[index]
  #      # Load HR image and apply HR transformation
   #     img = Image.open(os.path.join(self.path, img_path)).convert("RGB")
    #    img_hr = self.hr_transform(img)
        
     #   # Load corresponding LR image
      #  lr_image_path = "/workspace/storage/low_quality_images/" + img_path  # Replace "YOUR_LR_IMAGES_PATH" with the path to your LR images folder
       # lr_img = Image.open(lr_image_path).convert("RGB")
       # lr_img = transforms.ToTensor()(lr_img)
       # lr_img = transforms.Normalize(mean, std)(lr_img)

       # return {"lr": lr_img, "hr": img_hr}

   # def __len__(self):
    #    return len(self.files)

class ImageDataset(Dataset):
    def __init__(self, root, hr_shape,path = "HIGH_QUALITY_IMAGES"):
        hr_height, hr_width = hr_shape
        #df=pd.read_csv(csv_file, header=None)
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.path = path
        self.files = os.listdir(path)
        #self.file_names = list(df[0])
        #self.csv_file=df

    def __getitem__(self, index):
        img_path = self.files[index]
        img = Image.open(os.path.join(self.path, img_path)).convert('RGB')
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)
        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)
