import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import cv2 
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from PIL import Image

class CustomAIDetectionDataset(Dataset):
    def __init__(self, real_images, fake_images, transform=None):
        self.images = []
        self.labels = []
        for img in real_images:
            self.images.append(img)
            self.labels.append(0)
        for img in fake_images:
            self.images.append(img)
            self.labels.append(1)
           
        self.transform = transform
   
    def __len__(self):
        return len(self.images)
   
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
       
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
           
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
           
        image = image[np.newaxis, :, :]
       
        if self.transform:
            image = self.transform(image)
           
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)