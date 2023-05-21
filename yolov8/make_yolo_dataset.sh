#!/bin/bash

python utils/json2yolo_v2.py --coco_path="../hw3_data/hw3_dataset/org" --save_dir="./datasets/hw3_dataset_yolo"

python utils/yaml_generator_v2.py --coco_json_file="../hw3_data/hw3_dataset/org/train.coco.json" --data_path="datasets/hw3_dataset_yolo"  --out_path="./hw3_dataset_yolo.yaml" 