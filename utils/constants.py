# Description: Constants for the CNN model

# hyperparameters
batch_size = 32
nc = 3  #number of channels in the training images. For color images this is 3
ndf = 64  #size of feature maps propagated through the network
num_epochs = 5
lr = 0.0002
beta1 = 0.5
num_classes = 9
# num_classes = 3
seed = 2031998

#gesture classes
gesture_classes = ['dislike', 'fist', 'four', 'like', 'ok', 'one', 'palm', 'peace', 'three2']
# gesture_classes = ['paper', 'rock', 'scissors']

#pth
pth = '/Users/vale/Desktop/Sapienza/Vision/pth_folder/cnn_model_1.pth'      #funziona con 9 classi
# pth = '/Users/vale/Desktop/Sapienza/Vision/pth_folder/cnn_model_rps.pth'        #funziona con 3 classi

#yolo paths
config_path = 'yolo_hand_detection_master/models/cross-hands.cfg'
weights_path = 'yolo_hand_detection_master/models/cross-hands.weights'

#dataset paths
train_path = '/Users/vale/Desktop/Sapienza/Vision/Gestures&RPS/hand_poses_dataset_CROP'    #9 classes

# train_path = '/Users/vale/Desktop/Sapienza/Vision/Gestures&RPS/hand_poses_dataset_CROP_rps1'    #3 classes



test_path = '/Users/vale/Desktop/Sapienza/Vision/Gestures&RPS/hand_test_dataset_CROP'  #9 classes

# test_path = '/Users/vale/Desktop/Sapienza/Vision/Gestures&RPS/hand_test_dataset_CROP_RPS'  #3 classes


# TO RUN
# cd /Users/vale/Desktop/Sapienza/Vision/CNN_vision python -m test_and_train.test_cnn
# cd /Users/vale/Desktop/Sapienza/Vision/CNN_vision python -m test_and_train.train_cnn
# cd /Users/vale/Desktop/Sapienza/Vision/CNN_vision python -m webcam_run.webcam_mediapipe