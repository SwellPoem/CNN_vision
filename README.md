# CNN_vision

## Overview
This project develops a Convolutional Neural Network (CNN) designed to recognize and classify various hand gestures 👐🏻.

## Methodology
- **Data Collection:** The dataset used is the HaGRID dataset, simplified for the task representing 9 different hand gestures from a diverse group of individuals captured against varying backgrounds.
- **Model Architecture:** The CNN structure consists of several convolutional layers, batch normalization layers, ReLU activation layers, dropout layers, and fully connected layers.
- **Training Process:** The model was trained using an 80-20 split for training and test, over 5 epochs, employing the Adam optimizer and crossentropy as the loss function.

## Technologies Used
- Python
- PyTorch
- OpenCV

## Webcam
The CNN is used to perform real-time hand gesture recognition usign a webcam. 
The objective is achieved by the use of: a pre-trained YOLO (You Only Look Once) model for hand detection and the custom CNN for hand gesture classification.

## Additional Stuff
- **Rock Paper Scissors Game:** A variant of the simple real-time gesture recognition with the webcam. The user can play 'Rock, Paper, Scissors' against the network with a simple intereface.

- **TO DO: ASL detection**
