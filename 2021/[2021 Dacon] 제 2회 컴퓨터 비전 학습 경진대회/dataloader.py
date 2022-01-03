import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader 
from tqdm import tqdm

class PathDataset(Dataset): 
    def __init__(self, image_paths, labels=None, transforms=None, is_test=False): 
        self.image_paths = image_paths
        self.labels = labels 
        self.transforms = transforms
        self.is_test = is_test

        self.imgs = []

        for img_path in tqdm(self.image_paths):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.imgs.append(img)

    def __getitem__(self, index):
        # img = cv2.imread(self.image_paths[index])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.imgs[index]
        if self.transforms:
            img = self.transforms(image=img)['image'].astype(np.float32)
        img = self.normalize_img(img)
        img = self.to_torch_tensor(img)

        if self.is_test:
            return img
        else:
            return img, torch.tensor(self.labels[index], dtype=torch.float32)

    def __len__(self): 
        return len(self.image_paths)

    def normalize_img(self, img):
        mean = np.array([123.675, 116.28 , 103.53 ], dtype=np.float32)
        std = np.array([58.395   , 57.120, 57.375   ], dtype=np.float32)
        img = img.astype(np.float32)
        img -= mean
        img *= np.reciprocal(std, dtype=np.float32)
        return img

    def to_torch_tensor(self,img):
        return torch.from_numpy(img.transpose((2, 0, 1)))

