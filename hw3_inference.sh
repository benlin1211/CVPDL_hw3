#!/bin/bash 

# $1: testing images directory (e.g. input/test_dir/) that contains images in subdirectories
# $2: path of output json file (e.g. output/pred.json)
# $3: one of the numbers between [0] and [4], specifying which checkpoint to use, 
#     e.g., 0 indicates the 0% checkpoint, and 3 indicates the 100% checkpoint, and 4 indicates the best checkpoint.

# Eval
# python val.py --weight ./runs/train/exp/weights/best.pt

# Inference
if [[ $3 -eq 0 ]]
then
    echo "0% checkpoint loaded."
    weight="./R_YOLO/runs/train/exp/weights/epoch0.pt"
elif [[ $3 -eq 1 ]]
then
    echo "33% checkpoint loaded."
    weight="./R_YOLO/runs/train/exp/weights/epoch50.pt"
elif [[ $3 -eq 2 ]]
then
    echo "66% checkpoint loaded."
    weight="./R_YOLO/runs/train/exp/weights/epoch100.pt"
elif [[ $3 -eq 3 ]]
then
    echo "100% checkpoint loaded."
    weight="./R_YOLO/runs/train/exp/weights/epoch149.pt"
elif [[ $3 -eq 4 ]]
then
    echo "100% checkpoint loaded."
    weight="./R_YOLO/runs/train/exp/weights/best.pt"
else
    echo "Invalid argument, stopped."
fi

python ./R_YOLO/detect.py --weight $weight --source ${1} --output_path ${2}
# python ../check_your_prediction_valid.py ./output/pred_eval.json ../hw3_data/hw3_dataset/fog/val.coco.json 
