# HandGesture CNN

## Overview
This project develops a Convolutional Neural Network (CNN) designed to recognize and classify various hand gestures 👐🏻.

## Methodology
- **Data Collection:** The dataset used is the HaGRID dataset, simplified for the task representing 9 different hand gestures from a diverse group of individuals captured against varying backgrounds. ![alt text](https://github.com/SwellPoem/CNN_vision/blob/main/readme_images/IMG_0435.jpg)
- **Model Architecture:** The CNN structure consists of several convolutional layers, batch normalization layers, ReLU activation layers, dropout layers, and fully connected layers.
- **Training Process:** The model was trained using an 80-20 split for training and test, over 5 epochs, employing the Adam optimizer and crossentropy as the loss function. First the dataset was passed over to a pre-trained YOLO model for hand detection, in order to identify and crop all the detected hands in the images, and then the actual training of the HandGesture CNN was performed on the cropped dataset. The accuracy reached after the test process is of 95.36%.

## Technologies Used
- Python 3.12
- PyTorch
- OpenCV
- YOLO
- MediaPipe Hand Landmarker

## Webcam
The CNN is used to perform real-time hand gesture recognition usign a webcam. 
The objective is achieved by the use of: MediaPipe Hand Landmarker for hand landmarks detection and the custom CNN for hand gesture classification.

## Weights
The ``` .pth ``` files can be downloaded at this [link](https://drive.google.com/drive/folders/1WVqr217AOnhyXsT0e7ei5433wcgNX5w1?usp=sharing).

## Additional Stuff
- **Rock Paper Scissors Game:** A variant of the simple real-time gesture recognition with the webcam. The user can play 'Rock, Paper, Scissors' against the network with a simple intereface.

## Structure
``` bash
├── CNN_vision
    ├── readme_images
    │    └── image
    ├── test_and_train
    │    ├── test_cnn.py
    │    └── train_cnn.py
    ├── utils
    │    ├── constants.py
    │    ├── dataset_augment
    │    ├── dataset_converter.py
    │    ├── dataset.py
    │    ├── model.py
    │    └── preprocess.py
    ├── webcam_run
    │    ├── webcam_mediapipe.py
    │    ├── webcam_rps.py
    │    └── webcam.py
    ├── .gitignore
    ├── cnn.py
    └── README.md
```

## How to run
- Download the repo
- Locate in the folder of the download
- To run the simple hand detection with the webcam: in the terminal use the command ``` python -m webcam_run.webcam_mediapipe ```

- To run the Rock, Paper, Scissors game: in the terminal use the command ``` python -m webcam_run.webcam_rps ```
