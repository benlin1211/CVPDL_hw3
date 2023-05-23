# CVPDL_hw3
## Envirement Setup:
    # Create conda env:
    conda create -n R_YOLO python=3.10
    # Activate the enviorment:
    conda activate R_YOLO
    # Install:
    pip install -r requirements.txt
    
## Dataset download (if necessary):
    bash download_data.sh
    
## Checkpoint download:
    bash hw3_download.sh

## Run training code:
    bash hw3_train.sh $1 $2 $3 $4 $5
Parameters:
- $1: train source images directory. (./hw3_data/hw3_dataset/org/train/)
- $2: train target images directory. (./hw3_data/hw3_dataset/fog/train/)
- $3: eval target images directory. ( ./hw3_data/hw3_dataset/fog/val/)
- $4: train source label path. (./hw3_data/hw3_dataset/org/train.coco.json)
- $5: eval target label path. (./hw3_data/hw3_dataset/fog/val.coco.json)    

e.g.
    
    bash hw3_train.sh ./hw3_data/hw3_dataset/org/train/ ./hw3_data/hw3_dataset/fog/train/ ./hw3_data/hw3_dataset/fog/val/ ./hw3_data/hw3_dataset/org/train.coco.json ./hw3_data/hw3_dataset/fog/val.coco.json

## Run evaluation code:
    python ./R_YOLO/val.py --weight ./R_YOLO/runs/train/exp/weights/best.pt

# Inference
    bash hw3_inference.sh $1 $2 $3
Parameters:
- $1: testing images directory that contains images in subdirectories. (e.g. input/test_dir/) 
- $2: path of output json file. (e.g. output/pred.json)
- $3: one of the numbers between 0 and 4, specifying which checkpoint to use.
      e.g., 0 indicates the 0% checkpoint, and 3 indicates the 100% checkpoint, and 4 indicates the best checkpoint.

e.g.

    bash hw3_inference.sh ./hw3_data_eval/hw3_dataset/ ./output/pred_eval.json 4
