import cv2
import os
import numpy as np
import logging
from tqdm import tqdm
from random import randint
from albumentations import RandomBrightnessContrast, Rotate, RandomResizedCrop, GaussianBlur
from imgaug import augmenters as iaa

#initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_transformations(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in tqdm(files, desc="Processing"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                image = cv2.imread(path)

                #1. Random zoom (crop) between 0% and 20%
                height, width, _ = image.shape
                crop_size = randint(0, min(height, width) // 2)
                x = randint(0, width - crop_size)
                y = randint(0, height - crop_size)
                cropped_image = image[y:y+crop_size, x:x+crop_size]
                crop = RandomResizedCrop(height, width, scale=(0.6, 1.0))  # scale between 80% (zoom in 20%) and 100% (no zoom)
                cropped_image = crop(image=image)['image']

                # 2. Random rotation between -20 and +20 degrees
                rotate = Rotate(limit=30)
                rotated_image = rotate(image=image)['image']

                # 3. Random brightness adjustment between -31% and +31%
                brightness = RandomBrightnessContrast(brightness_limit=0.31)
                bright_image = brightness(image=image)['image']


                # 5. Random cutout of 8 boxes with 11% size each
                cutout = iaa.Cutout(nb_iterations=(5, 10), size=0.15, squared=False, cval=0)
                cutout_image = cutout.augment_image(image)

                base_name, ext = os.path.splitext(file)
                output_file_cropped = f"{base_name}_cropped{ext}"
                output_file_rotated = f"{base_name}_rotated{ext}"
                output_file_bright = f"{base_name}_bright{ext}"
                output_file_cutout = f"{base_name}_cutout{ext}"

                output_path_cropped = os.path.join(output_folder, os.path.relpath(root, input_folder), output_file_cropped)
                output_path_rotated = os.path.join(output_folder, os.path.relpath(root, input_folder), output_file_rotated)
                output_path_bright = os.path.join(output_folder, os.path.relpath(root, input_folder), output_file_bright)
                output_path_cutout = os.path.join(output_folder, os.path.relpath(root, input_folder), output_file_cutout)

                if not os.path.exists(os.path.dirname(output_path_rotated)):
                    os.makedirs(os.path.dirname(output_path_rotated))

                cv2.imwrite(output_path_cropped, cropped_image)
                cv2.imwrite(output_path_rotated, rotated_image)
                cv2.imwrite(output_path_bright, bright_image)
                cv2.imwrite(output_path_cutout, cutout_image)

                logging.info(f"Processed and saved transformed images: {output_path_rotated}, {output_path_cutout}, {output_path_bright}")

def main(input_folder, output_folder):
    apply_transformations(input_folder, output_folder)
    logging.info("Completed processing all images.")

if __name__ == "__main__":
    input_folder = ""
    output_folder = ""
    main(input_folder, output_folder)