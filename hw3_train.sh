#!/bin/bash 
# bash hw3_train.sh ./hw3_data/hw3_dataset/org/train/ ./hw3_data/hw3_dataset/fog/train/ ./hw3_data/hw3_dataset/fog/val/ ./hw3_data/hw3_dataset/org/train.coco.json ./hw3_data/hw3_dataset/fog/val.coco.json

# $1: train source images directory
# $2: train target images directory
# $3: eval target images directory
# $4: train source label path (./**/train.coco.json)
# $5: eval target label path (./**/val.coco.json)

train_image_org=$1
train_image_adv=$2
val_images_adv=$3

train_json_path_org=$4
val_json_path_adv=$5

# prepare data
# bash ./R_YOLO/data_preprocesssing.sh
python ./R_YOLO/utils/move_data.py --png_dir $train_image_org --save_dir ./R_YOLO/dataset/Normal_to_Foggy/images --mode train --is_origin
python ./R_YOLO/utils/move_data.py --png_dir $train_image_adv --save_dir ./R_YOLO/dataset/Normal_to_Foggy/images --mode train 
python ./R_YOLO/utils/move_data.py --png_dir $val_images_adv --save_dir ./R_YOLO/dataset/Normal_to_Foggy/images --mode val 
## convert label
# origin train
python ./R_YOLO/utils/label_json2Ryolo.py --coco_file $train_json_path_org --save_dir ./dataset/Normal_to_Foggy/labels/ --mode train --is_origin
# adverse has no train
# python utils/label_json2Ryolo.py --coco_path ../hw3_data/hw3_dataset/fog/ --save_dir ./dataset/Normal_to_Foggy/labels/ --mode train
# adverse val
python ./R_YOLO/utils/label_json2Ryolo.py --coco_file $val_json_path_adv --save_dir ./dataset/Normal_to_Foggy/labels/ --mode val 


# Step 1
# train the model of normal_to_adverse
python ./R_YOLO/QTNet_train.py --mode normal_to_adverse --input_dir ./R_YOLO/dataset/Normal_to_Foggy/images/Normal_train/ \
                      --gt_dir ./R_YOLO/dataset/Normal_to_Foggy/images/Foggy_train/
# train the model of adverse_to_normal
python ./R_YOLO/QTNet_train.py --mode adverse_to_normal --input_dir ./R_YOLO/dataset/Normal_to_Foggy/images/Foggy_train/ \
                      --gt_dir ./R_YOLO/dataset/Normal_to_Foggy/images/Normal_train/

# generate the normal translation image 
python ./R_YOLO/QTNet_infer.py --mode normal_to_adverse --input_dir ./R_YOLO/dataset/Normal_to_Foggy/images/Normal_train/ \
                      --weight ./R_YOLO/runs/QTNet_weights/normal_to_foggy/_49.pth
# generate the adverse translation image 
python ./R_YOLO/QTNet_infer.py --mode adverse_to_normal --input_dir ./R_YOLO/dataset/Normal_to_Foggy/images/Foggy_train/ \
                      --weight ./R_YOLO/runs/QTNet_weights/foggy_to_normal/_49.pth

# generate adverse fake labels (generated from origin)
python ./R_YOLO/utils/label_json2Ryolo.py --coco_file $train_json_path_org --save_dir ./R_YOLO/dataset/Normal_to_Foggy/labels/ --mode train --is_origin --is_fake

# move the translation image
cp -r ./R_YOLO/dataset/Normal_to_Foggy/images/Foggy_fake/* ./R_YOLO/dataset/Normal_to_Foggy/images/Foggy_train/ 
cp -r ./R_YOLO/dataset/Normal_to_Foggy/images/Normal_fake/* ./R_YOLO/dataset/Normal_to_Foggy/images/Normal_train/ 

# --------------------------------
# Train:
python ./R_YOLOtrain_FCNet.py --data ./R_YOLO/hw3_train.yaml --cfg ./R_YOLO/yolov5m.yaml --weights ./R_YOLO/yolov5m.pt --batch-size 4 --img-size 2048 
# Resume: python train_FCNet.py --data hw3_train.yaml --cfg yolov5m.yaml --weights yolov5m.pt --batch-size 4 --img-size 2048 --resume
