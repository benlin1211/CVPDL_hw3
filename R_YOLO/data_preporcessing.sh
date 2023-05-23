#!/bin/bash

# ${1}: train images directory (../hw3_data/hw3_dataset/org/train/)
# ${2}: valid images directory (../hw3_data/hw3_dataset/fog/valid/)

# Convert data and label to acceptable dataset format.
## move data
python utils/move_data.py --png_dir ../hw3_data/hw3_dataset/org/train/ --save_dir ./dataset/Normal_to_Foggy/images --mode train --is_origin
python utils/move_data.py --png_dir ../hw3_data/hw3_dataset/fog/train/ --save_dir ./dataset/Normal_to_Foggy/images --mode train 
python utils/move_data.py --png_dir ../hw3_data/hw3_dataset/fog/val/ --save_dir ./dataset/Normal_to_Foggy/images --mode val 
## convert label
# origin train
python utils/label_json2Ryolo.py --coco_file ../hw3_data/hw3_dataset/org/train.coco.json --save_dir ./dataset/Normal_to_Foggy/labels/ --mode train --is_origin
# adverse has no train
# python utils/label_json2Ryolo.py --coco_path ../hw3_data/hw3_dataset/fog/ --save_dir ./dataset/Normal_to_Foggy/labels/ --mode train
# adverse val
python utils/label_json2Ryolo.py --coco_file ../hw3_data/hw3_dataset/fog/val.coco.json --save_dir ./dataset/Normal_to_Foggy/labels/ --mode val 


# # train the model of normal_to_adverse
# python QTNet_train.py --mode normal_to_adverse --input_dir ./dataset/Normal_to_Foggy/images/Normal_train/
#                       --gt_dir ./dataset/Normal_to_Foggy/images/Foggy_train/
# # train the model of adverse_to_normal
# python QTNet_train.py --mode adverse_to_normal --input_dir ./dataset/Normal_to_Foggy/images/Foggy_train/ \
#                       --gt_dir ./dataset/Normal_to_Foggy/images/Normal_train/
# Download my trained weight for QTNet:


# generate the normal translation image 
python QTNet_infer.py --mode normal_to_adverse --input_dir ./dataset/Normal_to_Foggy/images/Normal_train/ \
                      --weight ./runs/QTNet_weights/normal_to_foggy/_49.pth
# generate the adverse translation image 
python QTNet_infer.py --mode adverse_to_normal --input_dir ./dataset/Normal_to_Foggy/images/Foggy_train/ \
                      --weight ./runs/QTNet_weights/foggy_to_normal/_49.pth


# generate adverse fake labels (generated from origin)
python utils/label_json2Ryolo.py --coco_file ../hw3_data/hw3_dataset/org/train.coco.json --save_dir ./dataset/Normal_to_Foggy/labels/ --mode train --is_origin --is_fake


# move the translation image
cp -r ./dataset/Normal_to_Foggy/images/Foggy_fake/* ./dataset/Normal_to_Foggy/images/Foggy_train/ 
cp -r ./dataset/Normal_to_Foggy/images/Normal_fake/* ./dataset/Normal_to_Foggy/images/Normal_train/ 