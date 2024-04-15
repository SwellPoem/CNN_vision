import cv2
import torch
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from yolo_hand_detection_master.yolo import YOLO  # Make sure this import is correct
from cnn import HandGestureCNN  # Adjust this import based on your directory structure
from constants import *


# Instantiate the YOLO detector
yolo_config = config_path
yolo_weights = weights_path
yolo_labels = ['Hand']  # Assuming you have one class for 'Hand'
yolo_size = 416  # The size parameter must match what was used during YOLO training
yolo_detector = YOLO(yolo_config, yolo_weights, yolo_labels, size=yolo_size)

# Load your HandGestureCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
if device.type == 'cuda':
    print(f'Device selected: {torch.cuda.get_device_name(0)}')
    torch.cuda.manual_seed_all(seed)
else:
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        torch.mps.manual_seed_all(seed)
        print(x)
    else:
        print("MPS device not found.")

hand_gesture_cnn = HandGestureCNN(nc=nc, ndf=ndf, num_classes=num_classes).to(device)
hand_gesture_cnn.load_state_dict(torch.load(pth), map_location=device)
hand_gesture_cnn.eval()

# Seed 
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define the transformation
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
    # Convert the cropped hand numpy array to a PIL Image
    cropped_hand_pil = Image.fromarray(cropped_hand)
    # Apply the transformations
    preprocessed_hand = transform(cropped_hand_pil)
    # Add an extra batch dimension since pytorch expects batches of images
    preprocessed_hand = preprocessed_hand.unsqueeze(0).to(device)

    return preprocessed_hand

# Initialize webcam
cap = cv2.VideoCapture(0)

# Main loop to process the webcam feed
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference to detect hands
    width, height, inference_time, yolo_results = yolo_detector.inference(frame)

    # Display FPS
    cv2.putText(frame, f'{round(1/inference_time,2)} FPS', (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

    # Sort detections by confidence
    yolo_results.sort(key=lambda x: x[2], reverse=True)  # Assuming higher confidence first

    # Process each detection for gesture classification
    for detection in yolo_results:
        id, name, confidence, x, y, w, h = detection

        # Add a 5% margin to the bounding box
        margin = 0.05  # 5%
        x = int(x - w * margin)
        y = int(y - h * margin)
        w = int(w * (1 + 2 * margin))
        h = int(h * (1 + 2 * margin))

        # Ensure the bounding box is within the frame
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)

        # Crop the detected hand
        cropped_hand = frame[y:y+h, x:x+w]
        
        # Preprocess the cropped hand for HandGestureCNN
        preprocessed_hand = preprocess_for_hand_gesture_cnn(cropped_hand)
        
        # Run HandGestureCNN inference to classify the gesture
        gesture = "Unknown"  # Default if no prediction exceeds threshold
        with torch.no_grad():
            prediction = hand_gesture_cnn(preprocessed_hand)
            max_value, predicted_idx = torch.max(prediction, 1)
            if max_value.item() > 0.3:  # Use your confidence threshold
                gesture = gesture_classes[predicted_idx.item()]

        # Draw bounding box and label with gesture
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        text = f"{name} ({round(confidence, 2)}): {gesture}"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Show the modified frame
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()