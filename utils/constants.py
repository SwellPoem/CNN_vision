# hyperparameters
batch_size = 32
nc = 3  #number of channels in the training images. For color images this is 3
ndf = 64  #size of feature maps propagated through the network
num_epochs = 5
lr = 0.0002
beta1 = 0.5
# num_classes = 9
num_classes = 24
# num_classes = 3
seed = 2031998

#gesture classes
# gesture_classes = ['dislike', 'fist', 'four', 'like', 'ok', 'one', 'palm', 'peace', 'three2']
gesture_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
# gesture_classes = ['paper', 'rock', 'scissors']

#pth
# pth = '/Users/vale/Desktop/Sapienza/Vision/pth_folder/cnn_model_1.pth'      #funziona anche con mediapipe -> some problems fist
# pth = '/Users/vale/Desktop/Sapienza/Vision/pth_folder/cnn_model_rps.pth'        #funziona anche con mediapipe

pth = '/Users/vale/Desktop/Sapienza/Vision/pth_folder/cnn_model_asl_dataset_augmented.pth'


#yolo paths
config_path = 'yolo_hand_detection_master/models/cross-hands.cfg'
weights_path = 'yolo_hand_detection_master/models/cross-hands.weights'

#dataset paths
# train_path = '/Users/vale/Desktop/Sapienza/Vision/hand_poses_dataset_CROP'    #9 classes

# train_path = '/Users/vale/Desktop/Sapienza/Vision/Gestures&RPS/hand_poses_dataset_mediapipe_rps1'    #3 classes

train_path = '/Users/vale/Desktop/Sapienza/Vision/ASL/ASL_Dataset_part_AUGMENTED/Train'


# test_path = '/Users/vale/Desktop/Sapienza/Vision/Gestures&RPS/hand_test_dataset_CROP'  #9 classes

# test_path = '/Users/vale/Desktop/Sapienza/Vision/Gestures&RPS/hand_test_dataset_CROP_RPS'  #3 classes

test_path = '/Users/vale/Downloads/asl_dataset'

# TO RUN
# cd /Users/vale/Desktop/Sapienza/Vision/CNN_vision python -m test_and_train.test_cnn
# cd /Users/vale/Desktop/Sapienza/Vision/CNN_vision python -m test_and_train.train_cnn
# cd /Users/vale/Desktop/Sapienza/Vision/CNN_vision python -m webcam_run.webcam
# cd /Users/vale/Desktop/Sapienza/Vision/CNN_vision python -m webcam_run.webcam_rps
# cd /Users/vale/Desktop/Sapienza/Vision/CNN_vision python -m utils.dataset_converter