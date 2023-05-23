# For more args, please infer to https://docs.ultralytics.com/usage/cfg/#modes
# https://huggingface.co/keremberke/yolov8m-valorant-detection
# https://docs.ultralytics.com/modes/predict/#sources
import argparse
from ultralytics import YOLO
import numpy as np
import torch
import random
import glob
from PIL import Image
import os
import torchvision.transforms as transforms
import json
from tqdm import tqdm
import cv2
from pathlib import Path

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--seed', default=42, type=int) 
    # ===================== Train Config ====================
    parser.add_argument('--dont_freeze', action='store_true') 
    parser.add_argument('--epochs', default=400, type=int) 
    parser.add_argument('--batch', default=2, type=int)
    parser.add_argument('--lr0', default=1e-2, type=float)
    parser.add_argument('--lrf', default=1e-5, type=float)
    #  https://docs.ultralytics.com/modes/train/#arguments
    # ===================== Eval Config =====================
    parser.add_argument('--resume', default='', help='resume from checkpoint (xxx.pt)') #./runs/detect/train/weights/best.pt
    parser.add_argument('--confidence', default=0.25, type=float) 
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_path', default='./datasets/hw1_dataset_yolo/images/val', type=str)
    # ===================== Test Config =====================
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_path', default='./datasets/hw1_dataset_yolo/images/test', type=str)
    parser.add_argument('--out_path', default='./output/pred_test.json', type=str)
    parser.add_argument('--file_name', default='./all_feature.npy', type=str)
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--is_adverse', action='store_true') 
    
    return parser

def generate_result(model, data_path, confidence, prefix):
    results = {}
    
    data_path = f"{str(Path(data_path).resolve())}" + "/"
    print(data_path)

    # pbar = tqdm(glob.glob(os.path.join(data_path, "*.png")))
    pbar = tqdm(sorted(glob.glob(f"{data_path}/**/*.png", recursive=True)))
    for img_name in pbar:
        img = Image.open(img_name)
        pred = model(img, conf=args.confidence, verbose=False)[0]
        # print("res:",pred.boxes.xyxy, pred.boxes.cls, pred.boxes.conf)
        
        boxes = pred.boxes.xyxy.cpu().numpy()
        labels = pred.boxes.cls.cpu().numpy()
        scores = pred.boxes.conf.cpu().numpy()

        # do some operation
        # keep_idx = np.where(scores>=confidence)
        # boxes = boxes[keep_idx]
        # labels = labels[keep_idx]
        # scores = scores[keep_idx]    

        res = {} 
        res['boxes'] = boxes.tolist() 
        res['labels'] = labels.tolist() 
        res['scores'] = scores.tolist() 
        key_name = img_name.replace(data_path, "")
        # print(key_name)
        key_name = f'{prefix}/val/' + key_name.split("/")[-1]
        results[key_name] = res
    
    return results

def main(args):
    # Fix seed
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.eval:
        print(f"eval, resume from {args.resume}, confidence={args.confidence}")
        model = YOLO(args.resume)
        # https://docs.ultralytics.com/modes/predict/#sources
        # if args.is_adverse:
        #     metrics = model.val(data="./hw3_dataset_adverse.yaml", batch=16, save_json=True, conf=args.confidence)
        # else:
        #     metrics = model.val(data="./hw3_dataset_yolo.yaml", batch=16, save_json=True, conf=args.confidence)
        if args.is_adverse:
            prefix = "fog"
        else: 
            prefix = "org"
        results = generate_result(model, args.eval_path, args.confidence, prefix=prefix)
        out_dir = args.out_path.replace(args.out_path.split('/')[-1], "")
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out_path, 'w') as fp:
            json.dump(results, fp)
        print(f"Eval result is saved at {args.out_path}")
        return

    if args.test:
        print(f"test, resume from {args.resume}")
        model = YOLO(args.resume)
        results = generate_result(model, args.test_path, args.confidence)
        out_dir = args.out_path.replace(args.out_path.split('/')[-1], "")
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out_path, 'w') as fp:
            json.dump(results, fp)
        print(f"Test result is saved at {args.out_path}")
        #print("WIP")
        return
    
    if args.report:
        layer_selected = -1
        features = {}       
        def get_features(name):
            def hook(model, input, output):
                x, y = output
                # print(x.shape)
                features[name] = x.detach().cpu().numpy().flatten()
            return hook
        
        print(f"Report, resume from {args.resume}")
        model = YOLO(args.resume)

        # register the forward hook
        model.model.model[layer_selected].register_forward_hook(get_features('desired_feature_layer'))
        
        data_path = f"{str(Path(args.test_path).resolve())}" + "/"
        # pbar = tqdm(glob.glob(os.path.join(data_path, "*.png")))
        pbar = tqdm(sorted(glob.glob(f"{data_path}/**/*.png", recursive=True)))
        all_features = []
        for img_name in pbar:
            img = Image.open(img_name)
            pred = model(img, conf=args.confidence, verbose=False)[0]
            # print(features['desired_feature_layer'].shape)
            all_features.append(features['desired_feature_layer'])

        all_features = np.array(all_features)
        with open(args.file_name, 'wb') as f:
            np.save(f, all_features)
        print(f"SAved feature at {args.file_name}")
        return
    

    # Load a model
    if args.resume:
        model = YOLO(args.resume)
        # For more args, please infer to https://docs.ultralytics.com/usage/cfg/#modes
    else:  
        model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

    # num_freeze = trainer.args.
    if ~args.dont_freeze:
        def freeze_layer(trainer):
            model = trainer.model
            num_freeze = 0
            print(f"Freezing {num_freeze} layers")
            freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze 
            for k, v in model.named_parameters(): 
                v.requires_grad = True  # train all layers 
                if any(x in k for x in freeze): 
                    print(f'freezing {k}') 
                    v.requires_grad = False 
            print(f"{num_freeze} layers are freezed.")

        # Add the custom callback to the model
        model.add_callback("on_train_start", freeze_layer)

    # Train the model
    # https://github.com/ultralytics/ultralytics/issues/713
    model.train(data="./hw3_dataset_yolo.yaml", 
                imgsz=(1024, 2048),
                device="0",
                epochs=args.epochs, 
                batch=args.batch, 
                lr0=args.lr0, 
                lrf=args.lrf,
                box=7.5,  # box loss gain
                cls=0.5,  # BCE Loss gain (scale with pixels)
                dfl=1.0,    # Distribution Focal Loss gain, 原本1.5 # https://zhuanlan.zhihu.com/p/310229973
                # 
                close_mosaic=True,
                # data augmentation
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=0.0,
                translate=0.1,
                scale=0.5,
                shear=0.2,
                flipud=0.5,
                fliplr=0.5,
                mosaic=1.0,  # image mosaic (probability)
                mixup=0.2,  # image mixup (probability)
                )  # train the model

    # save the model to ONNX format
    success = model.export(format="onnx")  
    print(success)

    # metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
