# Description:
#This file contains the model architecture for the CNN model

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from cnn import HandGestureCNN
from utils.constants import *
from torchsummary import summary
from torchviz import make_dot

#instantiate the model
model = HandGestureCNN()

summary(model, input_size=(nc, 64, 64))
