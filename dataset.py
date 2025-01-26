from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
from PIL import Image
from utils import *

class Datasets(Dataset):
    def __init__(self, path_Data, datasets, train=True):
        super().__init__()
        if train:
            images_list = os.listdir(path_Data + 'train/images/')
            masks_list = os.listdir(path_Data + 'train/masks/')
            images_list = sorted(images_list)
            masks_list = sorted(masks_list)
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'train/images/' + images_list[i]
                mask_path = path_Data + 'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transform = transforms.Compose([
                myNormalize(datasets, train=True),
                myToTensor(),
                myRandomHorizontalFlip(p=0.5),
                myRandomVerticalFlip(p=0.5),
                myRandomRotation(p=0.5, degree=[0, 360]),
                myResize(256, 256)
            ])
        else:
            images_list = os.listdir(path_Data + 'val/images/')
            masks_list = os.listdir(path_Data + 'val/masks/')
            images_list = sorted(images_list)
            masks_list = sorted(masks_list)
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'val/images/' + images_list[i]
                mask_path = path_Data + 'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transform = transforms.Compose([
                myNormalize(datasets, train=True),
                myToTensor(),
                myRandomHorizontalFlip(p=0.5),
                myRandomVerticalFlip(p=0.5),
                myRandomRotation(p=0.5, degree=[0, 360]),
                myResize(256, 256)
            ])

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        mask = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, mask = self.transform((img, mask))
        return img, mask

    def __len__(self):
        return len(self.data)