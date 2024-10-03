# Description:
#This script uses the webcam to detect hand gestures using the MediaPipe Hands model
#and classifies them using the HandGestureCNN model.

import cv2
import mediapipe as mp
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
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
            cropped_hand_pil = Image.fromarray(cropped_hand)
            preprocessed_hand = transform(cropped_hand_pil)
            preprocessed_hand = preprocessed_hand.unsqueeze(0).to(device)

            # classify the gesture
            with torch.no_grad():
                prediction = hand_gesture_cnn(preprocessed_hand)
                max_value, predicted_idx = torch.max(prediction, 1)
                if max_value.item() > 0.3:  #threshold
                    gesture = gesture_classes[predicted_idx.item()]

            
            text = f"Hand: {gesture}"
            cv2.putText(frame, text, (int(x_min_abs), int(y_min_abs - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


