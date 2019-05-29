# Pytorch stuff
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch
import torch.utils.data as utils
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler


# Input 76x76 output 16x16
class AtrousSeg(nn.Module):
    def __init__(self, num_classes=1, num_channels=8):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(num_channels),
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, dilation = 1), # Front           
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, dilation = 2),            
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, dilation = 2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, dilation = 3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, dilation = 3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, dilation = 3),            
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, dilation = 3), #LFE
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation = 3), 
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation = 3), 
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation = 2), 
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation = 2), 
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation = 1), 
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation = 1),             
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 1024, kernel_size=7, stride=1, padding=1, dilation = 3), # Head (44x44)
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, dilation = 1), 
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, dilation = 1),
            #nn.Sigmoid(),
            nn.UpsamplingBilinear2d(size=(76, 76)),
        )
        # Initialize Weights
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        result = self.model(x)
        # Better for MSE
        #return result
        # Better for BCEWithLogitsLoss
        return torch.sigmoid(result)
        return F.sigmoid(result)
        #return result
        #return self.model(x)
        if self.training:
            return result
        else:
            return F.sigmoid(result)
        #    result = self.model(x)
        #    print(x.shape)
        #    print(result.shape)
        #    F.softmax(result, dim=2)