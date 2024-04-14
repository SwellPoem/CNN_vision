import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from cnn import HandGestureCNN

# Define transformations
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load the test dataset
test_dataset = datasets.ImageFolder(root='hand_test_dataset_cnn', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Device configuration
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

# Initialize your CNN model
num_classes = 18  # Define the number of classes
nc = 3  # Number of channels in the training images. For color images, this is 3
ndf = 64 
model = HandGestureCNN(nc, ndf, num_classes).to(device)

# Load the trained model
model.load_state_dict(torch.load('cnn_model2.pth', map_location=device))
model.eval()  # Set the model to evaluation mode

def evaluate_model(model, dataloader, device):
    print("Starting test process...")
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc="Testing")
    with torch.no_grad():  # No need to track gradients during evaluation
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

# Evaluate the model
evaluate_model(model, test_dataloader, device)
