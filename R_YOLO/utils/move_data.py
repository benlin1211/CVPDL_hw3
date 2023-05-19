# Ref: Please see 
# https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
# https://blog.csdn.net/llh_1178/article/details/114528795.
# # https://github.com/ultralytics/yolov3/issues/738
# # https://blog.csdn.net/IYXUAN/article/details/124339385
# and my hw1: https://github.com/benlin1211/CVPDL_hw1/blob/main/yolov8/utils/json2yolo.py
import json
from collections import defaultdict
import argparse
import os
import glob
from tqdm import tqdm
import numpy as np


def move_data(png_dir, save_dir, mode, is_origin):
    # Set up previx string
    if is_origin==True:
        prefix = "Normal"
    else:
        prefix = "Foggy"

    # Create output directory
    os.makedirs(save_dir, exist_ok=True) 
    
    save_image_dir = os.path.join(save_dir, f"{prefix}_{mode}")
    os.makedirs(save_image_dir, exist_ok=True)
    # print(save_image_dir)

    # copy image
    # print(png_dir, save_image_dir)
    os.system(f'cp -r {os.path.join(png_dir, "*")} {save_image_dir}')
    print("done")

# Convert xxx.coco.json and train/* to ./images and ./labels
if __name__ == '__main__':

    parser = argparse.ArgumentParser('Move data', add_help=False)
    parser.add_argument('--png_dir', default="") # directory of *.png
    parser.add_argument('--save_dir', default="") # directory to save ./images and ./labels
    #../dataset/Normal_to_Foggy/
    parser.add_argument('--mode', default="train", choices=['train', 'val', 'public_test']) 
    parser.add_argument('--is_origin', action='store_true') # origin or adverse (foggy) 

    args = parser.parse_args()
    # print(args)
        
    print(f"move_data from {args.png_dir} to {args.save_dir} ")
    move_data(args.png_dir,
              args.save_dir,
              args.mode,
              args.is_origin)

