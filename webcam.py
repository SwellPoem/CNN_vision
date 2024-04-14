import cv2
import torch
from torchvision import transforms
from PIL import Image
import torch
from yolo_hand_detection_master.yolo import YOLO
from cnn import HandGestureCNN
import numpy as np

# Set the seed for reproducibility
seed = 2031998
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Instantiate the YOLO detector
yolo_config = 'yolo_hand_detection_master/models/cross-hands.cfg'
yolo_weights = 'yolo_hand_detection_master/models/cross-hands.weights'
yolo_labels = ['Hand'] 
yolo_size = 416  # The size parameter must match what was used during YOLO training
yolo_detector = YOLO(yolo_config, yolo_weights, yolo_labels, size=yolo_size)

# Load HandGestureCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hand_gesture_cnn = HandGestureCNN(nc=3, ndf=64, num_classes=17).to(device)
hand_gesture_cnn.load_state_dict(torch.load('cnn_model.pth', map_location=torch.device('cpu')))
hand_gesture_cnn.eval()

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


gesture_classes = ['call', 'dislike', 'fist', 'four', 'like', 'ok', 'one', 'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three', 'three2', 'two_up', 'two_up_inverted']

# Initialize webcam
print("starting webcam...")
cv2.namedWindow("Webcam Feed")
cap = cv2.VideoCapture(0)

# Main loop to process the webcam feed
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference to detect hands
    width, height, inference_time, yolo_results = yolo_detector.inference(frame)

    # Display FPS
    cv2.putText(frame, f'{round(1/inference_time,2)} FPS', (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Sort detections by confidence
    yolo_results.sort(key=lambda x: x[2])

    # Define a margin percentage 
    margin_percentage = 0.05

    # Process each detection for gesture classification
    for detection in yolo_results:
        id, name, confidence, x, y, w, h = detection

        # Calculate margin in pixels
        margin_w = int(w * margin_percentage)
        margin_h = int(h * margin_percentage)

        # Apply margin to the bounding box coordinates, ensuring they stay within frame bounds
        x_expanded = max(x - margin_w, 0)
        y_expanded = max(y - margin_h, 0)
        w_expanded = min(frame.shape[1] - x_expanded, w + (2 * margin_w))
        h_expanded = min(frame.shape[0] - y_expanded, h + (2 * margin_h))

        # Crop and preprocess the detected hand for HandGestureCNN
        cropped_hand = frame[y_expanded:y_expanded+h_expanded, x_expanded:x_expanded+w_expanded]
        preprocessed_hand = preprocess_for_hand_gesture_cnn(cropped_hand)
        
        # Run HandGestureCNN inference to classify the gesture
        gesture = "Unknown"  # Default if no prediction exceeds threshold
        with torch.no_grad():
            prediction = hand_gesture_cnn(preprocessed_hand)
            max_value, predicted_idx = torch.max(prediction, 1)
            print(f"Max value: {max_value.item()}")
            if max_value.item() > 0.1:  # Use your confidence threshold
                gesture = gesture_classes[predicted_idx.item()]

        # Draw expanded bounding box and label with gesture
        cv2.rectangle(frame, (x_expanded, y_expanded), (x_expanded + w_expanded, y_expanded + h_expanded), (255, 0, 0), 2)
        text = f"{name} ({round(confidence, 2)}): {gesture}"
        cv2.putText(frame, text, (x_expanded, y_expanded - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the modified frame
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
