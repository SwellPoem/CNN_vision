#Description:
#This script evaluates the trained CNN model on the test dataset and computes the confusion matrix.

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from cnn import HandGestureCNN
from utils.constants import *
from sklearn.metrics import confusion_matrix


#transformations
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

#load test dataset
test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

#device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

if device.type == 'cuda':
    print(f'Device selected: {torch.cuda.get_device_name(0)}')
else:
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")

#seed 
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#CNN model
model = HandGestureCNN(nc, ndf, num_classes).to(device)

#load pre-trained model
model.load_state_dict(torch.load(pth, map_location=device))
model.eval()  #set the model to evaluation mode

def evaluate_model(model, dataloader, device, class_names):
    print("Starting test process...")
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    progress_bar = tqdm(dataloader, desc="Testing")
    with torch.no_grad():  #no need to track gradients during evaluation
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

    #confusion matrix computation
    cm = confusion_matrix(all_labels, all_predictions)

    #normalize
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='.1f', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

class_names = test_dataset.classes

evaluate_model(model, test_dataloader, device, class_names)

