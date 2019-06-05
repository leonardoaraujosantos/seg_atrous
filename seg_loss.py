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

# Reference: https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a
# https://arxiv.org/pdf/1606.04797v1.pdf
def dice_loss(model_outputs, labels, do_sigmoid=False):    
    smooth = 1e-4
    
    # Apply sigmoid if the model is NOT doing already
    if do_sigmoid:
        model_outputs = torch.sigmoid(model_outputs)

    iflat = model_outputs.contiguous().view(-1)
    tflat = labels.contiguous().view(-1)
    intersection =  torch.abs(iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              ((iflat*iflat).sum() + (tflat*tflat).sum() + smooth))


# model_outputs, labels format: BATCH x Channels x ROWs x COLs
# Reference: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
# This requires that both labels and model_outputs are on the same range (0..1)
def iou_loss(model_outputs, labels, do_sigmoid=False):
    smooth = 1e-4
    
    # Apply sigmoid if the model is NOT doing already
    if do_sigmoid:
        model_outputs = torch.sigmoid(model_outputs)
        
    # Avoid negative values 
    intersection = torch.abs(model_outputs * labels).sum()
    union = torch.abs(model_outputs).sum() + torch.abs(labels).sum()
    iou = (intersection + smooth) / (union - intersection + smooth)
    # Invert to "minimize" to just plug and play on the lost
    return 1 - iou