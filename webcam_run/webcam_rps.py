#Description: This script runs a Rock-Paper-Scissors game using the webcam feed.
#The user plays against the computer, which randomly selects a gesture.
#The user's gesture is determined by the hand gesture classification model.
#The game starts when the user presses the "Start" button and ends after 3 seconds
#when the user's gesture is classified and the winner is determined.
#The user can reset the game by pressing the "Reset" button.

import cv2
import torch
import mediapipe as mp
import numpy as np
import random
from torchvision import transforms
from PIL import Image
from cnn import HandGestureCNN
from utils.constants import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer

# Mediapipe initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.9)

# HandGestureCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hand_gesture_cnn = HandGestureCNN(nc=nc, ndf=ndf, num_classes=num_classes).to(device)
hand_gesture_cnn.load_state_dict(torch.load(pth, map_location=torch.device('cpu')))
hand_gesture_cnn.eval()

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

# Create a PyQt5 window with a label for the countdown and the random choice
app = QApplication([])
window = QMainWindow()
central_widget = QWidget()
layout = QVBoxLayout()

greeting_label = QLabel("Get ready to play!")
countdown_label = QLabel()
random_choice_label = QLabel()
greeting_label.setStyleSheet("font-size: 20px;")
countdown_label.setStyleSheet("font-size: 15px;")
layout.addWidget(greeting_label)
layout.addWidget(countdown_label)
layout.addWidget(random_choice_label)

# Add start and reset buttons
game_started = False

def start_game():
    global game_started
    game_started = True
    update_countdown(3)
    QTimer.singleShot(3000, update_random_choice)
    start_button.setEnabled(False)

def reset_game():
    global game_started
    game_started = False
    countdown_label.setText("")
    random_choice_label.setText("")
    winner_label.setText("")
    start_button.setEnabled(True)

start_button = QPushButton("Start")
start_button.setStyleSheet("background-color: green; color: white; font-size: 20px;")  # green background, white text, 20px font
start_button.clicked.connect(start_game)

reset_button = QPushButton("Reset")
reset_button.setStyleSheet("background-color: red; color: white; font-size: 20px;")  # red background, white text, 20px font
reset_button.clicked.connect(reset_game)

layout.addWidget(start_button)
layout.addWidget(reset_button)

central_widget.setLayout(layout)
window.setCentralWidget(central_widget)
window.show()

# Functions to update countdown and random choice
def get_computer_choice():
    return random.choice(gesture_classes)

def update_countdown(i):
    if i > 0:
        countdown_label.setText(str(i))
        QTimer.singleShot(1000, lambda: update_countdown(i-1))
    else:
        start_button.setEnabled(False)

winner_label = QLabel()
winner_label.setStyleSheet("font-size: 20px;")
layout.addWidget(winner_label)

def update_random_choice():
    if game_started:
        computer_choice = get_computer_choice()
        random_choice_label.setText(f"Computer's choice: {computer_choice}")
        winner = determine_winner(gesture, computer_choice)  # assuming 'gesture' is the user's choice
        winner_label.setText(f"{winner}")


update_countdown(3)
update_random_choice()


def determine_winner(user_choice, computer_choice):
    if user_choice == computer_choice:
        return "Draw"
    if (user_choice == "rock" and computer_choice == "scissors") or \
       (user_choice == "scissors" and computer_choice == "paper") or \
       (user_choice == "paper" and computer_choice == "rock"):
        return "User wins"
    return "Computer wins"


# Webcam initialization
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

    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Start the PyQt5 mainloop
app.exec_()