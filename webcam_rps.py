import cv2
import torch
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from yolo_hand_detection_master.yolo import YOLO
from cnn import HandGestureCNN
from constants import *
import random
import time
import tkinter as tk
from tkinter import StringVar
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer

# YOLO detector
yolo_config = config_path
yolo_weights = weights_path
yolo_labels = ['Hand']  # assuming to have one class for 'Hand'
yolo_size = 416  # the size parameter must match what was used during YOLO training
yolo_detector = YOLO(yolo_config, yolo_weights, yolo_labels, size=yolo_size)

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

# Call these functions before the main loop
update_countdown(3)
update_random_choice()

# Preprocess the cropped hand image for HandGestureCNN inference.
def preprocess_for_hand_gesture_cnn(cropped_hand):
    cropped_hand_pil = Image.fromarray(cropped_hand)    # convert the cropped hand numpy array to a PIL Image
    preprocessed_hand = transform(cropped_hand_pil)     # apply the transformations
    preprocessed_hand = preprocessed_hand.unsqueeze(0).to(device)   # add an extra batch dimension since pytorch expects batches of images

    return preprocessed_hand


def determine_winner(user_choice, computer_choice):
    if user_choice == computer_choice:
        return "Draw"
    if (user_choice == "rock" and computer_choice == "scissors") or \
       (user_choice == "scissors" and computer_choice == "paper") or \
       (user_choice == "paper" and computer_choice == "rock"):
        return "User wins"
    return "Computer wins"


# webcam initialization
cap = cv2.VideoCapture(0)

# main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference to detect hands
    width, height, inference_time, yolo_results = yolo_detector.inference(frame)

    # FPS
    cv2.putText(frame, f'{round(1/inference_time,2)} FPS', (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

    # sort detections by confidence
    yolo_results.sort(key=lambda x: x[2], reverse=True)  # higher confidence first

    # gesture classification for each detected hand
    for detection in yolo_results:
        id, name, confidence, x, y, w, h = detection

        # 5% more margin to the bounding box
        margin = 0.05
        x = int(x - w * margin)
        y = int(y - h * margin)
        w = int(w * (1 + 2 * margin))
        h = int(h * (1 + 2 * margin))

        # ensuring the bounding box is within the frame
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)

        # crop detected hand
        cropped_hand = frame[y:y+h, x:x+w]

        # preprocess the cropped hand for HandGestureCNN
        preprocessed_hand = preprocess_for_hand_gesture_cnn(cropped_hand)

        # run CNN inference to classify the gesture
        gesture = "Unknown"  # if no prediction exceeds threshold
        with torch.no_grad():
            prediction = hand_gesture_cnn(preprocessed_hand)
            max_value, predicted_idx = torch.max(prediction, 1)
            if max_value.item() > 0.3:  # threshold
                gesture = gesture_classes[predicted_idx.item()]

        # draw bounding box and label with gesture
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        text = f"{name} ({round(confidence, 2)}): {gesture}"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # show frame
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

# Start the PyQt5 mainloop
app.exec_()