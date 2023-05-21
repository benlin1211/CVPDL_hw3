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


def convert_coco_json(coco_file, save_dir, mode, is_origin, is_fake):

    # Set up previx string
    if is_origin==True:
        prefix = "Normal"
    else:
        prefix = "Foggy"

    # Create output directory
    os.makedirs(save_dir, exist_ok=True) 

    # ========================== Convert label ==========================
    if not os.path.isfile(coco_file):
        print(f"{coco_file} not found.")
    else:
        # Create label directory with prefix
        save_label_dir = os.path.join(save_dir, f"{prefix}_{mode}")
        os.makedirs(save_label_dir, exist_ok=True)

        with open(coco_file) as f:
            data = json.load(f)
            
        # Create image dict
        images = {'%g' % x['id']: x for x in data['images']}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data['annotations']:
            imgToAnns[ann['image_id']].append(ann)

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f'Annotations {coco_file}'):
            img = images['%g' % img_id]
            # Retrieve data fields 
            h, w, f = img['height'], img['width'], img['file_name']

            bboxes = []
            for ann in anns:
                if ann['iscrowd']:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                #cls = coco80[ann['category_id'] - 1] if cls91to80 else ann['category_id'] - 1  # class
                cls = ann['category_id']
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)

            # Write
            txt_name = f.split("/")[-1].split(".")[0]

            if is_fake: 
                final_name = os.path.join(save_label_dir, f"source_{txt_name}_fake_B.txt")
            else: 
                final_name = os.path.join(save_label_dir, f"{txt_name}.txt")
            # print(final_name)
            with open(final_name, 'w') as file:
                # print("Write")
                for i in range(len(bboxes)):
                    line = *(bboxes[i]),  # cls, box or segments
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')
        print("done")


# Convert xxx.coco.json and train/* to ./images and ./labels
if __name__ == '__main__':

    parser = argparse.ArgumentParser('Coco to yolo label', add_help=False)
    parser.add_argument('--coco_file', default="") # directory of *.json
    #../../hw3_data/hw3_dataset/org/
    #../../hw3_data/hw3_dataset/fog/
    parser.add_argument('--save_dir', default="") # directory to save ./images and ./labels
    #../dataset/Normal_to_Foggy/
    parser.add_argument('--mode', default="train", choices=['train', 'val', 'public_test']) 
    parser.add_argument('--is_origin', action='store_true') # origin or adverse (foggy) 
    parser.add_argument('--is_fake', action='store_true') # real or fake

    args = parser.parse_args()
    # print(args)
    convert_coco_json(args.coco_file,  
                      args.save_dir,
                      args.mode,
                      args.is_origin,
                      args.is_fake)


    
