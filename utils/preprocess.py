import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm

# initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def preprocess_dataset(dataset_path, output_path, img_size=(64, 64)):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    img_paths = list(dataset_path.glob('**/*.jpg'))
    for img_path in tqdm(img_paths, desc="Processing images"):
        img = cv2.imread(str(img_path))
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # create an empty heatmap
                heatmap = np.zeros(img_size)

                # add each landmark to the heatmap
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * img_size[1]), int(landmark.y * img_size[0])
                    x = np.clip(x, 0, img_size[1] - 1)
                    y = np.clip(y, 0, img_size[0] - 1) 
                    heatmap[y, x] = 1  # set the pixel at the landmark's position to 1

                # save the heatmap to a file in the output directory
                output_file_path = output_path / img_path.relative_to(dataset_path).with_suffix('.jpg')
                output_file_path.parent.mkdir(parents=True, exist_ok=True)  # create subdirectories if they don't exist
                cv2.imwrite(str(output_file_path), heatmap * 255)  # multiply by 255 to get pixel values in the range [0, 255]


preprocess_dataset("/Users/vale/Desktop/Sapienza/Vision/ASL/ASL_prova/Train", "/Users/vale/Desktop/Sapienza/Vision/ASL/ASL_prova_mediapipe/Train")