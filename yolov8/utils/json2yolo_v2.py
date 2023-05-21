# Please see 
# https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
# https://blog.csdn.net/llh_1178/article/details/114528795.
# # https://github.com/ultralytics/yolov3/issues/738
# # https://blog.csdn.net/IYXUAN/article/details/124339385
import json
from collections import defaultdict
import argparse
import os
import glob
from tqdm import tqdm
import numpy as np

def create_img_folder(json_dir, prefix, save_dir):
    image_dir = os.path.join(save_dir, 'images', prefix)
    os.makedirs(image_dir, exist_ok=True)
    os.system(f'cp -r {json_dir}/* {image_dir}')
    # for image in sorted(glob.glob(os.path.join(json_dir, '*.png'))):
    #     # print(image)
    #     os.system(f'cp {image} {image_dir}')


def convert_coco_json(root_dir, prefix, save_dir):
    print(f"source dir: {root_dir}, mode: {prefix}, saved at {save_dir}")
    # output directory
    os.makedirs(save_dir, exist_ok=True) 

    # copy image
    image_source = os.path.join(root_dir, prefix)
    create_img_folder(image_source, prefix, save_dir)

    # Create label 
    save_json_dir = os.path.join(save_dir, 'labels', prefix)
    os.makedirs(save_json_dir, exist_ok=True)
    print(save_json_dir)
    # for coco_file in sorted(glob.glob(os.path.join(json_dir, '*.json'))):
    coco_file = os.path.join(root_dir, f'{prefix}.coco.json')

    if not os.path.isfile(coco_file):
        print(f"{coco_file} not found.")
    else:
        with open(coco_file) as f:
            data = json.load(f)

        # Create image dict
        images = {'%g' % x['id']: x for x in data['images']}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data['annotations']:
            imgToAnns[ann['image_id']].append(ann)

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f'Annotations {coco_file}'):
            img = images['%g' % img_id]
            h, w, f = img['height'], img['width'], img['file_name']

            bboxes = []
            for ann in anns:
                if ann['iscrowd']:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                #cls = coco80[ann['category_id'] - 1] if cls91to80 else ann['category_id'] - 1  # class
                cls = ann['category_id']
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)

            # Write
            # print(os.path.join(save_json_dir, f"{f}.txt"))
            txt_name = f.split("/")[-1].split(".")[0]
            with open(os.path.join(save_json_dir, f"{txt_name}.txt"), 'w') as file:
                # print("Write")
                for i in range(len(bboxes)):
                    line = *(bboxes[i]),  # cls, box or segments
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--coco_path', default="") #../../hw1_dataset
    parser.add_argument('--save_dir', default="") #../datasets/hw1_dataset_yolo_train
    parser.add_argument('--adverse_path', default=None) 
    # parser.add_argument('--save_dir_train', default="") #../datasets/hw1_dataset_yolo_train
    # parser.add_argument('--save_dir_valid', default="") #../datasets/hw1_dataset_yolo_valid

    args = parser.parse_args()

    convert_coco_json(args.coco_path,  # directory with *.json
                      'train',
                      args.save_dir)
    if args.adverse_path is not None:
        convert_coco_json(args.adverse_path,  # directory with *.json
                        'val',
                        args.save_dir)        
    else:
        convert_coco_json(args.coco_path,  # directory with *.json
                        'val',
                        args.save_dir)
    
    # # There is no .json in test folder
    # create_img_folder(os.path.join(args.coco_path, "test") , "test", args.save_dir)

    # zip results
    # os.system('zip -r ../coco.zip ../coco')
    print("to yolo done")
