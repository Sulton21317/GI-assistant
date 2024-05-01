
# resize_images_512x512.py

# 1 This splits the original Kvasir-SEG: A Segmented Polyp Dataset 
# to three subsets train, test and valid. 
# https://paperswithcode.com/dataset/kvasir-seg

# 2 Resize all image and masks to 512x512
#

#    
import sys
import os
import glob
from PIL import Image, ImageOps
import random
import shutil
import traceback
import cv2
import numpy as np

# 2024/01/22
W = 512  # 256
H = 512  # 256


def resize_images(images_dir, masks_dir, output_dir):
    pattern = images_dir + "/*.jpg"

    print("--- pattern {}".format(pattern))
    image_files = glob.glob(pattern)
    num_files = len(image_files)
    # 1 shuffle mask_files
    random.shuffle(image_files)

    # 2 Compute the number of images to split
    # train= 0.8 test=0.2 
    num_train = int(num_files * 0.7)
    num_valid = int(num_files * 0.2)
    num_test = int(num_files * 0.1)

    train_files = image_files[0: num_train]
    valid_files = image_files[num_train: num_train + num_valid]
    test_files = image_files[num_train + num_valid:]

    print("=== number of train_files {}".format(len(train_files)))
    print("=== number of valid_files {}".format(len(valid_files)))
    print("=== number of test_files  {}".format(len(test_files)))

    # 3 Resize images and masks 
    create_resized_files(train_files, masks_dir, output_dir, "train")
    create_resized_files(valid_files, masks_dir, output_dir, "valid")
    # 4 If test generated_dataset, save the original file to output + "test" dir.

    create_resized_files(test_files, masks_dir, output_dir, "test")


def create_resized_files(image_files, masks_dir, output_dir, dataset):
    # target = train_or_test_or_valid_dir:

    output_subdir = os.path.join(output_dir, dataset)
    if os.path.exists(output_subdir):
        shutil.rmtree(output_subdir)

    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    output_images_dir = os.path.join(output_subdir, "images")
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    output_masks_dir = os.path.join(output_subdir, "masks")
    if not os.path.exists(output_masks_dir):
        os.makedirs(output_masks_dir)

    for image_file in image_files:
        basename = os.path.basename(image_file)

        nameonly = basename.split(".")[0]
        mask_file = os.path.join(masks_dir, basename)
        image = Image.open(image_file).convert("RGB")

        mask = Image.open(mask_file).convert("L")
        """
        if generated_dataset == "test":
          shutil.copy2(image_file, output_images_dir)
          shutil.copy2(mask_file,  output_masks_dir)
          continue
    
        else:
        """
        w, h = image.size
        size = w
        if h >= w:
            size = h
        px = int((size - w) / 2)
        py = int((size - h) / 2)

        image_background = Image.new("RGB", (size, size), (0, 0, 0))
        image_background.paste(image, (px, py))

        mask_background = Image.new("L", (size, size))
        mask_background.paste(mask, (px, py))
        resized_image = image_background.resize((W, H))
        resized_mask = mask_background.resize((W, H))

        ANGLES = [0, 90, 180, 270]

        for angle in ANGLES:
            rotated_image = resized_image.rotate(angle)
            rotated_mask = resized_mask.rotate(angle)
            output_filename = "rotated_" + str(angle) + "_" + basename

            rotated_image_file = os.path.join(output_images_dir, output_filename)
            rotated_image.save(rotated_image_file)
            rotated_mask_file = os.path.join(output_masks_dir, output_filename)
            rotated_mask.save(rotated_mask_file)

        # flipp
        flipped_image = ImageOps.flip(resized_image)
        flipped_mask = ImageOps.flip(resized_mask)
        output_filename = "flipped_" + basename

        flipped_image_file = os.path.join(output_images_dir, output_filename)
        flipped_image.save(flipped_image_file)
        flipped_mask_file = os.path.join(output_masks_dir, output_filename)
        flipped_mask.save(flipped_mask_file)

        # mirror
        mirrored_image = ImageOps.mirror(resized_image)
        mirrored_mask = ImageOps.mirror(resized_mask)

        output_filename = "mirrored_" + basename

        mirrored_image_file = os.path.join(output_images_dir, output_filename)
        mirrored_image.save(mirrored_image_file)
        mirrored_mask_file = os.path.join(output_masks_dir, output_filename)
        mirrored_mask.save(mirrored_mask_file)


"""
input_dir
Kvasir-SEG
 +-- images
 +-- masks

ouput_directory struture

"./generated_dataset"
 +-- test
 |     +--images
 |     +--mask
 +-- train
 |     +--images
 |     +--mask
 +-- valid
       +--images
       +--mask

 """

if __name__ == "__main__":
    try:
        images_dir = "./Kvasir-SEG/images"
        masks_dir = "./Kvasir-SEG/masks"
        output_dir = "generated_dataset"
        if not os.path.exists(images_dir):
            raise Exception("===NOT FOUND " + images_dir)
        if not os.path.exists(masks_dir):
            raise Exception("===NOT FOUND " + masks_dir)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # create_master_512x512 generated_dataset train, test
        # from the orignal SEG .
        resize_images(images_dir, masks_dir, output_dir)

    except:
        traceback.print_exc()

# %%
