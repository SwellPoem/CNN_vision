# Description:
#This file contains the model architecture for the CNN model

from torchsummary import summary
from cnn import HandGestureCNN
from utils.constants import *


#instantiate the model
model = HandGestureCNN()

summary(model, input_size=(nc, 64, 64))
