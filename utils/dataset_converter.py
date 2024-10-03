# Description:
#This script is used to crop hands from images using YOLOv3 model.

import cv2
import os
import numpy as np
import logging
from tqdm import tqdm

#initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(cfg_path, weights_path):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    return net

def expand_bounding_box(x, y, w, h, width, height, expansion_rate=0.05):
    """Expand the bounding box by a specified rate, ensuring it remains within image bounds."""
    new_w = int(w * (1 + expansion_rate))
    new_h = int(h * (1 + expansion_rate))
    
    #calculating the difference to adjust the top-left corner
    delta_w = new_w - w
    delta_h = new_h - h
    
    #adjusting the top-left corner to keep the box centered
    new_x = max(x - delta_w // 2, 0)
    new_y = max(y - delta_h // 2, 0)
    
    #ensuring the new bounding box does not exceed image dimensions
    new_x = min(new_x, width - new_w)
    new_y = min(new_y, height - new_h)
    
    return new_x, new_y, new_w, new_h

def detect_and_crop(input_folder, output_folder, net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    for root, dirs, files in os.walk(input_folder):
        for file in tqdm(files, desc="Processing"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                image = cv2.imread(path)
                height, width = image.shape[:2]

                blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:  #to filter out low-confidence detections
                            center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            #expand the bounding box
                            x, y, w, h = expand_bounding_box(x, y, w, h, width, height)

                            #ensure the bounding box is fully within the image dimensions
                            x, y, w, h = max(x, 0), max(y, 0), min(w, width), min(h, height)

                            #check if crop dimensions are valid
                            if w > 0 and h > 0:
                                cropped_image = image[y:y+h, x:x+w]

                                #construct output path with unique names for multiple detections
                                base_name, ext = os.path.splitext(file)
                                output_file = f"{base_name}_{x}_{y}{ext}"
                                output_path = os.path.join(output_folder, os.path.relpath(root, input_folder), output_file)

                                #create directory structure if it doesn't exist
                                if not os.path.exists(os.path.dirname(output_path)):
                                    os.makedirs(os.path.dirname(output_path))

                                #ensure cropped_image is not empty
                                if cropped_image.size > 0:
                                    cv2.imwrite(output_path, cropped_image)
                                    logging.info(f"Processed and saved hand crop: {output_path}")
                                else:
                                    logging.warning(f"Cropped image is empty for {path}")
                            else:
                                logging.warning(f"Invalid crop dimensions for {path}")

def main(input_folder, output_folder, cfg_path, weights_path):
    net = load_model(cfg_path, weights_path)
    detect_and_crop(input_folder, output_folder, net)
    logging.info("Completed processing all images.")

if __name__ == "__main__":
    input_folder = ""
    output_folder = ""
    cfg_path = "/Users/vale/Desktop/Sapienza/Vision/CNN_vision/yolo_hand_detection_master/models/cross-hands.cfg"
    weights_path = "/Users/vale/Desktop/Sapienza/Vision/CNN_vision/yolo_hand_detection_master/models/cross-hands.weights"
    main(input_folder, output_folder, cfg_path, weights_path)