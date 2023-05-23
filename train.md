
# Train
## Dataset download (if necessary):
    bash download_data.sh

## Envirement Setup:
    # Create conda env:
    conda create -n R_YOLO python=3.10
    # Activate the enviorment:
    conda activate R_YOLO
    # Install:
    pip install -r requirements.txt


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
