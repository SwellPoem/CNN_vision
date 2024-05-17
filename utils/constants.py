# hyperparameters
batch_size = 32
nc = 3  #number of channels in the training images. For color images this is 3
ndf = 64  #size of feature maps propagated through the network
num_epochs = 5
lr = 0.0002
beta1 = 0.5
# num_classes = 9
num_classes = 26
# num_classes = 3
seed = 2031998

#gesture classes
# gesture_classes = ['dislike', 'fist', 'four', 'like', 'ok', 'one', 'palm', 'peace', 'three2']
gesture_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# gesture_classes = ['paper', 'rock', 'scissors']

#pth
# pth = '/Users/vale/Desktop/Sapienza/Vision/pth_folder/cnn_model_1.pth'
# pth = '/Users/vale/Desktop/Sapienza/Vision/pth_folder/cnn_model_rps.pth'
# pth = '/Users/vale/Desktop/Sapienza/Vision/pth_folder/cnn_model_alphabet.pth'
pth = '/Users/vale/Desktop/Sapienza/Vision/pth_folder/cnn_model_alphabet_2.pth'


#yolo paths
config_path = 'yolo_hand_detection_master/models/cross-hands.cfg'
weights_path = 'yolo_hand_detection_master/models/cross-hands.weights'

#dataset paths
# train_path = '/Users/vale/Desktop/Sapienza/Vision/hand_poses_dataset_CROP'    #9 classes

# train_path = '/Users/vale/Desktop/Sapienza/Vision/hand_poses_dataset_CROP_rps'    #3 classes

# train_path = '/Users/vale/Desktop/Sapienza/Vision/archive/asl_alphabet_train/asl_alphabet_train'  #26 classes
train_path = '/Users/vale/Desktop/Sapienza/Vision/asl_dataset'


# test_path = '/Users/vale/Desktop/Sapienza/Vision/hand_test_dataset_CROP'  #9 classes

# test_path = '/Users/vale/Desktop/Sapienza/Vision/hand_test_dataset_CROP_RPS'  #3 classes

# test_path = '/Users/vale/Desktop/Sapienza/Vision/archive_blackwhite/Test' #26 classes
test_path = '/Users/vale/Desktop/Sapienza/Vision/asl_dataset'

# TO RUN
# cd /Users/vale/Desktop/Sapienza/Vision/CNN_vision python -m test_and_train.test_cnn
# cd /Users/vale/Desktop/Sapienza/Vision/CNN_vision python -m test_and_train.train_cnn
# cd /Users/vale/Desktop/Sapienza/Vision/CNN_vision python -m webcam_run.webcam
# cd /Users/vale/Desktop/Sapienza/Vision/CNN_vision python -m webcam_run.webcam_rps
# cd /Users/vale/Desktop/Sapienza/Vision/CNN_vision python -m utils.dataset_converter