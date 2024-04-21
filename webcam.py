import cv2
import torch
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from yolo_hand_detection_master.yolo import YOLO
from cnn import HandGestureCNN
from constants import *


#YOLO detector
yolo_config = config_path
yolo_weights = weights_path
yolo_labels = ['Hand']  #assuming to have one class for 'Hand'
yolo_size = 416  #the size parameter must match what was used during YOLO training
yolo_detector = YOLO(yolo_config, yolo_weights, yolo_labels, size=yolo_size)

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

def preprocess_for_hand_gesture_cnn(cropped_hand):
    """
    Preprocess the cropped hand image for HandGestureCNN inference.

    Parameters:
    cropped_hand (numpy.ndarray): The cropped hand region from the frame.

    Returns:
    torch.Tensor: The preprocessed image tensor.
    """
    cropped_hand_pil = Image.fromarray(cropped_hand)    #convert the cropped hand numpy array to a PIL Image
    preprocessed_hand = transform(cropped_hand_pil)     #apply the transformations
    preprocessed_hand = preprocessed_hand.unsqueeze(0).to(device)   #add an extra batch dimension since pytorch expects batches of images

    return preprocessed_hand

#webcam initialization
cap = cv2.VideoCapture(0)

#main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    #YOLO inference to detect hands
    width, height, inference_time, yolo_results = yolo_detector.inference(frame)

    #FPS
    cv2.putText(frame, f'{round(1/inference_time,2)} FPS', (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

    #sort detections by confidence
    yolo_results.sort(key=lambda x: x[2], reverse=True)  #higher confidence first

    #gesture classification for each detected hand
    for detection in yolo_results:
        id, name, confidence, x, y, w, h = detection

        #5% more margin to the bounding box
        margin = 0.05
        x = int(x - w * margin)
        y = int(y - h * margin)
        w = int(w * (1 + 2 * margin))
        h = int(h * (1 + 2 * margin))

        #ensuring the bounding box is within the frame
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)

        #crop detected hand
        cropped_hand = frame[y:y+h, x:x+w]
        
        #preprocess the cropped hand for HandGestureCNN
        preprocessed_hand = preprocess_for_hand_gesture_cnn(cropped_hand)
        
        #run CNN inference to classify the gesture
        gesture = "Unknown"  #if no prediction exceeds threshold
        with torch.no_grad():
            prediction = hand_gesture_cnn(preprocessed_hand)
            max_value, predicted_idx = torch.max(prediction, 1)
            if max_value.item() > 0.3:  #threshold
                gesture = gesture_classes[predicted_idx.item()]

        #draw bounding box and label with gesture
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        text = f"{name} ({round(confidence, 2)}): {gesture}"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    #show frame
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# import cv2
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2
# import numpy as np

# MARGIN = 10  # pixels
# FONT_SIZE = 1
# FONT_THICKNESS = 1
# HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# # Create an HandLandmarker object.
# base_options = python.BaseOptions(model_asset_path='/Users/vale/Desktop/Sapienza/Vision/hand_landmarker.task')
# options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
# detector = vision.HandLandmarker.create_from_options(options)

# def draw_landmarks_on_image(rgb_image, detection_result):
#   hand_landmarks_list = detection_result.hand_landmarks
#   handedness_list = detection_result.handedness
#   annotated_image = np.copy(rgb_image)

#   # Loop through the detected hands to visualize.
#   for idx in range(len(hand_landmarks_list)):
#     hand_landmarks = hand_landmarks_list[idx]
#     handedness = handedness_list[idx]

#     # Draw the hand landmarks.
#     hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#     hand_landmarks_proto.landmark.extend([
#       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
#     ])
#     solutions.drawing_utils.draw_landmarks(
#       annotated_image,
#       hand_landmarks_proto,
#       solutions.hands.HAND_CONNECTIONS,
#       solutions.drawing_styles.get_default_hand_landmarks_style(),
#       solutions.drawing_styles.get_default_hand_connections_style())

#     # Get the top left corner of the detected hand's bounding box.
#     height, width, _ = annotated_image.shape
#     x_coordinates = [landmark.x for landmark in hand_landmarks]
#     y_coordinates = [landmark.y for landmark in hand_landmarks]
#     text_x = int(min(x_coordinates) * width)
#     text_y = int(min(y_coordinates) * height) - MARGIN

#     # Draw handedness (left or right hand) on the image.
#     cv2.putText(annotated_image, f"{handedness[0].category_name}",
#                 (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
#                 FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

#   return annotated_image

# # Initialize the webcam feed.
# cap = cv2.VideoCapture(0)

# # Main loop.
# while True:
#     # Capture frame-by-frame.
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert the BGR image to RGB.
#     rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Create a MediaPipe Image object from the RGB image.
#     image = mp.Image.create_from_np(rgb_image)

#     # Detect hand landmarks from the input image.
#     detection_result = detector.detect(image)

#     # Process the classification result. In this case, visualize it.
#     annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

#     # Display the resulting frame.
#     cv2.imshow('Webcam Feed', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

#     # Break the loop on 'q' key press.
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything is done, release the capture.
# cap.release()
# cv2.destroyAllWindows()