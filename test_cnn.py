import cv2
import torch
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from yolo_hand_detection_master.yolo import YOLO
from cnn import HandGestureCNN
import numpy as np
from constants import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# YOLO detector
yolo_config = config_path
yolo_weights = weights_path
yolo_labels = ['Hand']  # assuming to have one class for 'Hand'
yolo_size = 416  # the size parameter must match what was used during YOLO training
yolo_detector = YOLO(yolo_config, yolo_weights, yolo_labels, size=yolo_size)

# HandGestureCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
if device.type == 'cuda':
    print(f'Device selected: {torch.cuda.get_device_name(0)}')
else:
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")

model = HandGestureCNN(nc=nc, ndf=ndf, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(pth, map_location=torch.device('cpu')))
model.eval()

# seed
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# transformation
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

def preprocess_for_hand_gesture_cnn(cropped_hand):
    cropped_hand_pil = Image.fromarray(cropped_hand)    # convert the cropped hand numpy array to a PIL Image
    preprocessed_hand = transform(cropped_hand_pil)     # apply the transformations
    preprocessed_hand = preprocessed_hand.unsqueeze(0).to(device)   # add an extra batch dimension since pytorch expects batches of images

    return preprocessed_hand

def evaluate_model(model, dataloader, device):
    print("Starting test process...")
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc="Testing")
    with torch.no_grad():  # No need to track gradients during evaluation
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Apply YOLO hand detection
            for i in range(images.shape[0]):
                image = images[i].permute(1, 2, 0).cpu().numpy()
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # YOLO inference to detect hands
                width, height, inference_time, yolo_results = yolo_detector.inference(image)
                
                # sort detections by confidence
                yolo_results.sort(key=lambda x: x[2], reverse=True)  # higher confidence first
                
                # crop detected hand
                for detection in yolo_results:
                    id, name, confidence, x, y, w, h = detection
                    cropped_hand = image[y:y+h, x:x+w]
                    image = preprocess_for_hand_gesture_cnn(cropped_hand)
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = torch.from_numpy(image).permute(2, 0, 1).to(device)
                images[i] = image

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

evaluate_model(model, test_dataloader, device)




# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import numpy as np
# from cnn import HandGestureCNN
# from constants import *


# #transformations
# transform = transforms.Compose([
#     transforms.Resize(64),
#     transforms.CenterCrop(64),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

# #load test dataset
# test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# #device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# if device.type == 'cuda':
#     print(f'Device selected: {torch.cuda.get_device_name(0)}')
# else:
#     if torch.backends.mps.is_available():
#         mps_device = torch.device("mps")
#         x = torch.ones(1, device=mps_device)
#         print(x)
#     else:
#         print("MPS device not found.")

# #seed 
# torch.manual_seed(seed)
# np.random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# #CNN model
# model = HandGestureCNN(nc, ndf, num_classes).to(device)

# #load pre-trained model
# model.load_state_dict(torch.load(pth, map_location=device))
# model.eval()  # Set the model to evaluation mode

# def evaluate_model(model, dataloader, device):
#     print("Starting test process...")
#     correct = 0
#     total = 0
#     progress_bar = tqdm(dataloader, desc="Testing")
#     with torch.no_grad():  # No need to track gradients during evaluation
#         for images, labels in progress_bar:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
            
#     accuracy = 100 * correct / total
#     print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

# evaluate_model(model, test_dataloader, device)

