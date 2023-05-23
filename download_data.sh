#!/bin/bash 

# Download dataset
gdown 1F1N_YgB_43Z-IQOwocmsjnBL3AjycrkC -O hw3_data.zip

# Unzip the downloaded zip file
mkdir hw3_data
unzip ./hw3_data.zip -d hw3_data
# rm -rf ./hw3_data.zip

# # Download vgg pre-trained model to prepare the data of translation image
# gdown 199luoCcfhAF_8kydAwziOIPVqyiLECbN -O ./R-YOLO/runs/vgg16_caffe.pth


## bash download_data.sh