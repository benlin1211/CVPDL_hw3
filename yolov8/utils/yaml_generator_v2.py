import json
import os
import yaml
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--coco_json_file', default="") #../../hw1_dataset/train/_annotations.coco.json
    parser.add_argument('--data_path', default="") #./hw1_dataset_yolo
    parser.add_argument('--out_path', default="") #../hw1_dataset_yolo.yaml
    parser.add_argument('--num_classes', default=8+1, type=int) # include background!
    # parser.add_argument('--is_adverse', action="store_true")
    # parser.add_argument('--freeze', default=50, type=int) 
    args = parser.parse_args()

    result = {}
    root = os.getcwd()
    result['path'] = os.path.join(root, args.data_path) # dataset 路径
    result['train'] = 'images/train' #os.path.join( args.data_path,'images/train')  # 相对 path 的路径
    result['val'] = 'images/val' # same as train.
    result['test'] = 'images/test' # None
    result['nc'] = args.num_classes
    # result['freeze'] = args.freeze
    result['names'] = {}


    if not os.path.isfile(args.coco_json_file):
        print(f"{args.coco_json_file} not found.")
    else:
        with open(args.coco_json_file) as f:
            data = json.load(f)
        
        # Add background class
        result['names'][0] = "background"
        for category in data["categories"]:
            # print(category["id"])
            # print(category["name"])
            #id = category["id"]
            _id = category["id"] 
            name = category["name"]
            result['names'][_id] = name

        with open(args.out_path,"w") as file:
            yaml.dump(result, file, sort_keys=False)

    # Classes
    # print(result['names'])
    print("yaml done")
