# Description: 
#This script runs the webcam feed and performs hand detection using YOLO 
#and gesture classification using HandGestureCNN.
#slow framerate
#Needs the yolo_hand_detection_master folder in the same directory as this script

import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from yolo_hand_detection_master.yolo import YOLO
from cnn import HandGestureCNN
from utils.constants import *


#YOLO detector
yolo_config = config_path
yolo_weights = weights_path
yolo_labels = ['Hand']  #assuming to have one class for 'Hand'
yolo_size = 416  #the size parameter must match what was used during YOLO training
yolo_detector = YOLO(yolo_config, yolo_weights, yolo_labels, size=yolo_size)

#HandGestureCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hand_gesture_cnn = HandGestureCNN(nc=nc, ndf=ndf, num_classes=num_classes).to(device)
hand_gesture_cnn.load_state_dict(torch.load(pth, map_location=torch.device('cpu')))
hand_gesture_cnn.eval()

#seed 
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#transformation
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def preprocess_for_hand_gesture_cnn(cropped_hand):
    """
    Preprocess the cropped hand image for HandGestureCNN inference.

    Parameters:
    cropped_hand (numpy.ndarray): The cropped hand region from the frame.

    Returns:
    torch.Tensor: The preprocessed image tensor.
    """
    cropped_hand_pil = Image.fromarray(cropped_hand)    #convert the cropped hand numpy array to a PIL Image
    preprocessed_hand = transform(cropped_hand_pil)     #apply the transformations
    preprocessed_hand = preprocessed_hand.unsqueeze(0).to(device)   #add an extra batch dimension since pytorch expects batches of images

    return preprocessed_hand

#webcam initialization
cap = cv2.VideoCapture(0)

#main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    #YOLO inference to detect hands
    width, height, inference_time, yolo_results = yolo_detector.inference(frame)

    #FPS
    cv2.putText(frame, f'{round(1/inference_time,2)} FPS', (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

    #sort detections by confidence
    yolo_results.sort(key=lambda x: x[2], reverse=True)  #higher confidence first

    #gesture classification for each detected hand
    for detection in yolo_results:
        id, name, confidence, x, y, w, h = detection

        #5% more margin to the bounding box
        margin = 0.05
        x = int(x - w * margin)
        y = int(y - h * margin)
        w = int(w * (1 + 2 * margin))
        h = int(h * (1 + 2 * margin))

        #ensuring the bounding box is within the frame
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)

        #crop detected hand
        cropped_hand = frame[y:y+h, x:x+w]
        
        #preprocess the cropped hand for HandGestureCNN
        preprocessed_hand = preprocess_for_hand_gesture_cnn(cropped_hand)
        
        #run CNN inference to classify the gesture
        gesture = "Unknown"  #if no prediction exceeds threshold
        with torch.no_grad():
            prediction = hand_gesture_cnn(preprocessed_hand)
            max_value, predicted_idx = torch.max(prediction, 1)
            if max_value.item() > 0.3:  #threshold
                gesture = gesture_classes[predicted_idx.item()]

        #draw bounding box and label with gesture
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        text = f"{name} ({round(confidence, 2)}): {gesture}"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    #show frame
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()