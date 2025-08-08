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

class FractalUnit(nn.Module):
    
    def __init__(self, input_channel=32, hidden_channel=32):
        super(FractalUnit, self).__init__()
        
        self.branch_conv = nn.Conv2d(input_channel,hidden_channel,kernel_size=3,padding=1)
        self.fusion_conv = nn.Conv2d(input_channel,hidden_channel,kernel_size=3,padding=1)
        
        self.activation = nn.LeakyReLU(0.2)
        self.norm = nn.InstanceNorm2d(hidden_channel)
        
    def split_into_branches(self, x):
        
        b,c,h,w = x.shape
        mid_h, mid_w = h //2 , w // 2 
        
        branch_00 = x[:,:,:mid_h,:mid_w]
        branch_01 = x[:,:,:mid_h,mid_w:]
        branch_10 = x[:,:,mid_h:,:mid_w]
        branch_11 = x[:,:,mid_h:,mid_w:]
        
        return [branch_00,branch_01,branch_10,branch_11]
    
    def forward(self,x): 
        
        branches = self.split_into_branches(x)
        
        processed_branches = []
        for branch in branches:
            processed = self.branch_conv(branch) #Apply convolution
            processed = self.norm(processed) #Apply normalization
            processed = self.activation(processed) #Apply LeakyRELu
            processed_branches.append(processed)
            
        min_h = min(b.shape[2] for b in processed_branches)
        min_w = min(b.shape[3] for b in processed_branches)
        
        resized_branches = []
        for branch in processed_branches:
            if branch.shape[2] != min_h or branch.shape[3] != min_w:
                resized = F.interpolate(branch, size=(min_h,min_w), mode='bilinear', align_corners=False)
            else:
                resized = branch
            resized_branches.append(resized)
                
        fused = resized_branches[0]
        for branch in resized_branches[1:]:
            fused = fused * branch
        
        output = self.fusion_conv(fused)
        output = self.norm(output)
        output = self.activation(output)
        
        next_level_input = torch.mean(torch.stack(resized_branches), dim=0)
        
        return output, next_level_input
            
            
class FractalCNN(nn.Module):
    
    def __init__(self, num_fractal_levels=3, hidden_channels=32, num_classes=2): 
        super(FractalCNN, self).__init__()
        
        self.num_fractal_levels= num_fractal_levels
        self.hidden_channels = hidden_channels
        
        self.spatial_conv1 = nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1)
        self.spatial_conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        
        self.fractal_units = nn.ModuleList([
            FractalUnit(hidden_channels, hidden_channels)
            for _ in range(num_fractal_levels)
        ])
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        classifier_input_size = hidden_channels * num_fractal_levels
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size,128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        self.activation = nn.LeakyReLU(0.2)
        self.norm = nn.InstanceNorm2d(hidden_channels)
        
    def extract_noise_residual_torch(self,x): 
        kernel_size = 7 
        padding = kernel_size // 2
        
        blurred = F.avg_pool2d(x, kernel_size, stride=1, padding=padding)
        noise_residual = x - blurred
        
        return noise_residual
    
    def compute_fft_magnitude(self,x):
        fft = torch.fft.fft2(x)
        fft_shifted = torch.fft.fftshift(fft,dim=(-2,-1))
        magnitude = torch.abs(fft_shifted)
        
        batch_size = magnitude.shape[0]
        for i in range(batch_size):
            max_val = torch.max(magnitude[i])
            
            if max_val > 0:
                magnitude[i] = magnitude[i] / max_val
                
        return magnitude
    def forward(self ,x ):
        batch_size = x.shape[0]
        noise_residual = self.extract_noise_residual_torch(x)
        
        spectrum = self.compute_fft_magnitude(noise_residual)
        
        features = self.spatial_conv1(spectrum)
        features = self.norm(features)
        features = self.activation(features)
        
        features = self.spatial_conv2(features)
        features = self.norm(features)
        features = self.activation(features)
        
        level_features = []
        current_spectrum = features 
        
        for level, fractal_unit in enumerate(self.fractal_units):
            
            if current_spectrum.shape[2] < 4 or current_spectrum.shape[3] < 4:
                print(f"Stopping at level {level}: spectrum too small {current_spectrum}")
                break
            
            level_output, next_level_input = fractal_unit(current_spectrum)
            
            level_pooled = self.global_pool(level_output)
            level_features.append(level_pooled.view(batch_size, -1))
            
            current_spectrum = F.avg_pool2d(next_level_input, 2 , stride=2 )
            
        if level_features:
            combined_features = torch.cat(level_features,dim=1)
        else:
            
            combined_features = self.global_pool(features).view(batch_size, -1)
            expected_size = self.hidden_channels * self.num_fractal_levels
            if combined_features.shape[1] < expected_size:
                padding = expected_size - combined_features.shape[1]
                combined_features = F.pad(combined_features, (0, padding))
                
        output = self.classifier(combined_features)
        return output
     