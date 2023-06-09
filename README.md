# CVPDL_hw3
## Envirement Setup:
    # Create conda env:
    conda creat -y -n R_YOLO python=3.10
    # Activate the enviorment:
    conda activate R_YOLO
    # Install:
    pip install -r requirements.txt
    
# Inference

## Checkpoint download:
    bash hw3_download.sh
    
## Run inference code:
    bash hw3_inference.sh $1 $2 $3
Parameters:
- $1: testing images directory that contains images in subdirectories. (e.g. input/test_dir/) 
- $2: path of output json file. (e.g. output/pred.json)
- $3: one of the numbers between 0 and 4, specifying which checkpoint to use.
      e.g., 0 indicates the 0% checkpoint, and 3 indicates the 100% checkpoint, and 4 indicates the best checkpoint.

e.g.

    bash hw3_inference.sh ./input/test_dir/ ./output/pred.json 4
    # bash hw3_inference.sh ./hw3_data_test/hw3_dataset/ ./output/pred_test.json 4
    # bash hw3_inference.sh ../../hw3_data_test/hw3_dataset/ ./output/pred_test.json 4
    # bash hw3_inference.sh ../../hw3_data_eval_test/ ./output/pred_mix.json 4

# Train
## Dataset download (if necessary):
    bash download_data.sh

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

    # Eval
    python val.py --weight ./runs/train/exp/weights/best.pt

## Run Inference code:
Inference on val:

    # bash hw3_inference.sh ./hw3_data_eval/hw3_dataset/ ./output/pred_eval.json 4
    python detect.py --weight ./runs/train/exp/weights/best.pt --source ../hw3_data_eval/hw3_dataset/ --output_path ./output/pred_eval.json
    python ./check_your_prediction_valid.py ./output/pred_eval.json ./hw3_data/hw3_dataset/fog/val.coco.json 

Inference on public_test:

    # bash hw3_inference.sh ./hw3_data_test/hw3_dataset/ ./output/pred_test.json 4
    python detect.py --weight ./runs/train/exp/weights/best.pt --source ../hw3_data_test/hw3_dataset/ --output_path ./output/pred_test.json

## Plot map@50
    python draw_plot.py --csv_file ./runs/train/exp/results.csv --out_file ./map50_R_YOLO.png
________________________
# For problem one:

## Envirement Setup:
    cd yolov8
    conda create --name yolov8 python=3.10
    conda activate yolov8
    pip install -r requirements.txt
    bash make_yolo_dataset.sh 
    bash make_adverse_dataset.sh 

## Train:
    python main.py

## Train from your own checkpoints
    python main.py --resume ./runs/detect/train/weights/last.pt

## Eval on origin
    python main.py --resume ./runs/detect/train/weights/last.pt --eval_path ../hw3_data/hw3_dataset/org/val --eval --out_path ./output/pred_eval.json
    python ../check_your_prediction_valid.py ./output/pred_eval.json ../hw3_data/hw3_dataset/org/val.coco.json 

## Eval on adverse
    python main.py --resume ./runs/detect/train/weights/last.pt --eval_path ../hw3_data/hw3_dataset/fog/val --eval --is_adverse --out_path ./output/pred_adverse.json
    python ../check_your_prediction_valid.py ./output/pred_adverse.json ../hw3_data/hw3_dataset/fog/val.coco.json

## Test 
    python main.py --resume ./runs/detect/train/weights/last.pt --test --test_path ../hw3_data_test/hw3_dataset/ --out_path ./output/pred_test.json

## Plot map@50

    python draw_plot.py --csv_file ./runs/detect/train/results.csv --out_file ./map50_yolov8.png
_______________
# Report 
## For report 1 adapted model: 
    cd ./R_YOLO
    python report.py --weight ./runs/train/exp/weights/best.pt --source ../hw3_data/hw3_dataset/org/val --file_name ./all_feature_3.npy --imgsz 640
    python report.py --weight ./runs/train/exp/weights/best.pt --source ../hw3_data/hw3_dataset/fog/val --file_name ./all_feature_4.npy --imgsz 640
    cd ..
    python report_tsne.py ./R_YOLO/all_feature_3.npy ./R_YOLO/all_feature_4.npy "t-SNE: Adapted model inference" ./report_bonus34.png


## For report 1 source model:
    cd ./yolov8 
    python main.py --resume ./runs/detect/train/weights/last.pt --report --test_path ../hw3_data/hw3_dataset/org/val --file_name ./all_feature_1.npy
    python main.py --resume ./runs/detect/train/weights/last.pt --report --test_path ../hw3_data/hw3_dataset/fog/val --file_name ./all_feature_2.npy
    cd ..
    python report_tsne.py ./yolov8/all_feature_1.npy ./yolov8/all_feature_2.npy "t-SNE: Source model inference" ./report_bonus12.png


## For report 2:

### Adaptive

	cd ./R_YOLO

Origin

	python detect.py --weight ./runs/train/exp/weights/best.pt --source ../hw3_data_org/hw3_dataset/ --output_path ./output/pred_origin.json
	python ../check_your_prediction_valid.py ./output/pred_origin.json ../hw3_data/hw3_dataset/org/val.coco.json 

Adverse

	python detect.py --weight ./runs/train/exp/weights/best.pt --source ../hw3_data_eval/hw3_dataset/ --output_path ./output/pred_adverse.json
	python ../check_your_prediction_valid.py ./output/pred_adverse.json ../hw3_data/hw3_dataset/fog/val.coco.json 


### Source

	cd ./yolov8

Origin	

	python main.py --resume ./runs/detect/train/weights/last.pt --eval_path ../hw3_data/hw3_dataset/org/val --eval --out_path ./output/pred_origin.json
	python ../check_your_prediction_valid.py ./output/pred_origin.json ../hw3_data/hw3_dataset/org/val.coco.json

Adverse

	python main.py --resume ./runs/detect/train/weights/last.pt --eval_path ../hw3_data/hw3_dataset/fog/val --eval --is_adverse --out_path ./output/pred_adverse.json
	python ../check_your_prediction_valid.py ./output/pred_adverse.json ../hw3_data/hw3_dataset/fog/val.coco.json

### Init

Origin	


	python main.py --resume "init" --eval_path ../hw3_data/hw3_dataset/org/val --eval --out_path ./output/pred_init_origin.json
	python ../check_your_prediction_valid.py ./output/pred_init_origin.json ../hw3_data/hw3_dataset/org/val.coco.json

Adverse


	python main.py --resume "init" --eval_path ../hw3_data/hw3_dataset/fog/val --eval --is_adverse --out_path ./output/pred_init_adverse.json
	python ../check_your_prediction_valid.py ./output/pred_init_adverse.json ../hw3_data/hw3_dataset/fog/val.coco.json

### Init: Coco pretrain

Origin

	python main.py --resume "coco"  --eval_path ../hw3_data/hw3_dataset/org/val --eval --out_path ./output/pred_coco_origin.json
	python ../check_your_prediction_valid.py ./output/pred_coco_origin.json ../hw3_data/hw3_dataset/org/val.coco.json

Adverse

	python main.py --resume "coco" --eval_path ../hw3_data/hw3_dataset/fog/val --eval --is_adverse --out_path ./output/pred_coco_adverse.json
	python ../check_your_prediction_valid.py ./output/pred_coco_adverse.json ../hw3_data/hw3_dataset/fog/val.coco.json
