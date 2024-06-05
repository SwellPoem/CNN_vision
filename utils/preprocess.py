# Description:
#This script preprocesses the hand poses dataset by adding landmarks to the images.

import cv2
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

# initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

def preprocess_dataset(dataset_path, output_path):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    img_paths = list(dataset_path.glob('**/*.png')) + list(dataset_path.glob('**/*.jpg')) + list(dataset_path.glob('**/*.jpeg'))
    pbar = tqdm(total=len(img_paths), desc="Processing images")
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # save the image with landmarks to a file in the output directory
            output_file_path = output_path / img_path.relative_to(dataset_path).with_suffix('.jpg')
            output_file_path.parent.mkdir(parents=True, exist_ok=True)  # create subdirectories if they don't exist
            cv2.imwrite(str(output_file_path), img)
        
        pbar.update(1)

    pbar.close()

preprocess_dataset("/Users/vale/Desktop/Sapienza/Vision/Gestures&RPS/hand_poses_dataset_CROP", "/Users/vale/Desktop/Sapienza/Vision/Gestures&RPS/hand_poses_dataset_mediapipe")