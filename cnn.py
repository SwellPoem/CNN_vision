import torch
import torch.nn as nn
import torch.nn.functional as F

class HandGestureCNN(nn.Module):
    def __init__(self, nc=3, ndf=64, num_classes=17):
        super(HandGestureCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(ndf * 8, ndf * 16, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(ndf * 16, ndf * 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(ndf * 32, ndf * 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Flatten()
        )
        self.fc1 = nn.Linear(ndf * 64, 256)  # First fully connected layer, increased size
        self.fc2 = nn.Linear(256, 128)  # Additional fully connected layer
        self.fc3 = nn.Linear(128, num_classes)  # Output layer
        self.dropout = nn.Dropout(0.5)  # Dropout layer to prevent overfitting
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dropout(x)  # Apply dropout after convolutional layers
        x = F.relu(self.fc1(x))  # Apply ReLU activation function
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))  # Apply ReLU activation function
        x = self.fc3(x)  # No activation here, nn.CrossEntropyLoss will apply softmax
        # x = self.softmax(x)
        return x

