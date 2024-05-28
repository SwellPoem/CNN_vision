import cv2
import mediapipe as mp
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from yolo_hand_detection_master.yolo import YOLO
from cnn import HandGestureCNN
from utils.constants import *

# Mediapipe initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.9)

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

# webcam initialization
cap = cv2.VideoCapture(0)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # convert the BGR image to RGB and process it with MediaPipe Hands
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # get bounding box coordinates around the hand
            hand_landmarks_array = np.array([[landmark.x, landmark.y] for landmark in hand_landmarks.landmark])
            x_min, y_min = hand_landmarks_array.min(axis=0)
            x_max, y_max = hand_landmarks_array.max(axis=0)

            # convert relative coordinates to absolute coordinates
            h, w, _ = frame.shape
            x_min_abs, x_max_abs = x_min * w, x_max * w
            y_min_abs, y_max_abs = y_min * h, y_max * h
            
            # expand bounding box by 5%
            x_min_abs = max(0, x_min_abs - 0.05 * w)
            y_min_abs = max(0, y_min_abs - 0.05 * h)
            x_max_abs = min(w, x_max_abs + 0.05 * w)
            y_max_abs = min(h, y_max_abs + 0.05 * h)

            # draw bounding box
            cv2.rectangle(frame, (int(x_min_abs), int(y_min_abs)), (int(x_max_abs), int(y_max_abs)), (0, 255, 0), 2)

            # crop detected hand
            cropped_hand = frame[int(y_min_abs):int(y_max_abs), int(x_min_abs):int(x_max_abs)]

            # preprocess for HandGestureCNN
            # preprocessed_hand = cv2.resize(cropped_hand, (64, 64))
            cropped_hand_pil = Image.fromarray(cropped_hand)
            preprocessed_hand = transform(cropped_hand_pil)
            # preprocessed_hand = torch.from_numpy(preprocessed_hand).float().permute(2, 0, 1).unsqueeze(0).to(device)
            preprocessed_hand = preprocessed_hand.unsqueeze(0).to(device)

            # classify the gesture
            with torch.no_grad():
                prediction = hand_gesture_cnn(preprocessed_hand)
                max_value, predicted_idx = torch.max(prediction, 1)
                if max_value.item() > 0.3:  #threshold
                    gesture = gesture_classes[predicted_idx.item()]

            
            text = f"Hand: {gesture}"
            cv2.putText(frame, text, (int(x_min_abs), int(y_min_abs - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

# # Main loop
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert the BGR image to RGB and process it with MediaPipe Hands
#     results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     # Draw hand landmarks for each detected hand
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             # Get bounding box coordinates around the hand
#             hand_landmarks_array = np.array([[landmark.x, landmark.y] for landmark in hand_landmarks.landmark])
#             x_min, y_min = hand_landmarks_array.min(axis=0)
#             x_max, y_max = hand_landmarks_array.max(axis=0)

#             # Convert relative coordinates to absolute coordinates
#             h, w, _ = frame.shape
#             x_min_abs, x_max_abs = x_min * w, x_max * w
#             y_min_abs, y_max_abs = y_min * h, y_max * h
            
#             # Expand bounding box by 5%
#             x_min_abs = max(0, x_min_abs - 0.05 * w)
#             y_min_abs = max(0, y_min_abs - 0.05 * h)
#             x_max_abs = min(w, x_max_abs + 0.05 * w)
#             y_max_abs = min(h, y_max_abs + 0.05 * h)

#             # Draw bounding box around the detected hand
#             cv2.rectangle(frame, (int(x_min_abs), int(y_min_abs)), (int(x_max_abs), int(y_max_abs)), (0, 255, 0), 2)

#             # Crop detected hand
#             cropped_hand = frame[int(y_min_abs):int(y_max_abs), int(x_min_abs):int(x_max_abs)]

#             # Create an empty heatmap
#             heatmap = np.zeros((64, 64))

#             # Add each landmark to the heatmap
#             for landmark in hand_landmarks.landmark:
#                 x, y = int(landmark.x * 64), int(landmark.y * 64)
#                 x = np.clip(x, 0, 63)  # Clamp x to the range [0, 63]
#                 y = np.clip(y, 0, 63)  # Clamp y to the range [0, 63]
#                 heatmap[y, x] = 1  # Set the pixel at the landmark's position to 1

#             # Reshape the heatmap to match the input shape of the model
#             heatmap = heatmap.reshape((1, 64, 64, 1))

#             # Convert the heatmap to a PIL image
#             heatmap_pil = Image.fromarray(heatmap)

#             # Preprocess the heatmap for HandGestureCNN
#             preprocessed_hand = transform(heatmap_pil)
#             preprocessed_hand = preprocessed_hand.unsqueeze(0).to(device)

#             # Run CNN inference to classify the gesture
#             with torch.no_grad():
#                 prediction = hand_gesture_cnn(preprocessed_hand)
#                 max_value, predicted_idx = torch.max(prediction, 1)
#                 if max_value.item() > 0.3:  #threshold
#                     gesture = gesture_classes[predicted_idx.item()]

#             # Draw label with gesture
#             text = f"Hand: {gesture}"
#             cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


