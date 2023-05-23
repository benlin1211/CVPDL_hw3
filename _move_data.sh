#!/bin/bash 

mkdir ./R_YOLO/dataset
# mkdir ./R_YOLO/dataset/Normal_to_Foggy
# mkdir ./R_YOLO/dataset/Normal_to_Foggy/images
# mkdir ./R_YOLO/dataset/Normal_to_Foggy/images/Normal_train
# mkdir ./R_YOLO/dataset/Normal_to_Foggy/images/Foggy_train
# mkdir ./R_YOLO/dataset/Normal_to_Foggy/images/Foggy_val
# mkdir ./R_YOLO/dataset/Normal_to_Foggy/labels
# mkdir ./R_YOLO/dataset/Normal_to_Foggy/labels/Normal_train
# mkdir ./R_YOLO/dataset/Normal_to_Foggy/labels/Foggy_val

# images
cp -r ./hw3_data/hw3_dataset/org/train/* ./R_YOLO/dataset/Normal_to_Foggy/images/Normal_train/
cp -r ./hw3_data/hw3_dataset/fog/train/* ./R_YOLO/dataset/Normal_to_Foggy/images/Foggy_train/
cp -r ./hw3_data/hw3_dataset/fog/val/* ./R_YOLO/dataset/Normal_to_Foggy/images/Foggy_val/

# labels
#cp -r ./hw3_data/hw3_dataset/fog/val/* ./R_YOLO/dataset/Normal_to_Foggy/images/Normal_train/
#cp -r ./hw3_data/hw3_dataset/fog/val/* ./R_YOLO/dataset/Normal_to_Foggy/images/Foggy_val/
