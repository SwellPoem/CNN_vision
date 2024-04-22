# hyperparameters
batch_size = 32
nc = 3  #number of channels in the training images. For color images this is 3
ndf = 64  #size of feature maps in the discriminator
num_epochs = 5
lr = 0.0002
beta1 = 0.5
num_classes = 3
# num_classes = 9
# num_classes = 24
seed = 2031998

#gesture classes
# gesture_classes = ['dislike', 'fist', 'four', 'like', 'ok', 'one', 'palm', 'peace', 'three2']
# gesture_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
gesture_classes = ['paper', 'rock', 'scissors']

#pth
# pth = '/Users/vale/Desktop/Sapienza/Vision/pth_folder/cnn_model_1.pth'
# pth = '/Users/vale/Desktop/Sapienza/Vision/pth_folder/cnn_model_alphabet.pth'
# pth = '/Users/vale/Desktop/Sapienza/Vision/pth_folder/cnn_model_rps.pth'
pth = '/Users/vale/Desktop/Sapienza/Vision/pth_folder/cnn_model_rps_2.pth'

#yolo paths
config_path = 'yolo_hand_detection_master/models/cross-hands.cfg'
weights_path = 'yolo_hand_detection_master/models/cross-hands.weights'

#dataset paths
# train_path = '/Users/vale/Desktop/Sapienza/Vision/hand_poses_dataset_CROP'
# train_path = '/Users/vale/Desktop/Sapienza/Vision/archive/Train'
train_path = '/Users/vale/Desktop/Sapienza/Vision/hand_poses_dataset_CROP_rps'
# test_path = '/Users/vale/Desktop/Sapienza/Vision/hand_test_dataset_CROP'
test_path = '/Users/vale/Desktop/Sapienza/Vision/archive/Test'