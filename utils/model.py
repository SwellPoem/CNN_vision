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


# # Create a dummy input 
# input = torch.randn(2, nc, 64, 64).unsqueeze(0)

# # Forward pass through the model
# output = model(input)

# # Create a graph of the model
# dot = make_dot(output, params=dict(model.named_parameters()))

# # Save the graph to a file
# dot.format = 'png'
# dot.render(filename='model')