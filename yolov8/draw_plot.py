import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_args_parser():
    parser = argparse.ArgumentParser('Draw map at 50 plot.', add_help=False)
    # parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--csv_file', default="./runs/detect/train/results.csv") 
    parser.add_argument('--out_file', default="./map50_yolov8.png") 
    
    # python utils/draw_plot.py --csv_file ./runs/train/exp/results.csv
    # ===================== Train Config ====================
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    x = []
    y = []
    x_markers = []
    y_markers = []
    interval = 50
    with open(args.csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for i, row in enumerate(csv_reader):
            if i==0:
                continue
            else:
                x.append(int(row[0]))
                y.append(float(row[6]))
                if (i-1)%interval == 0:
                    x_markers.append(int(row[0]))
                    y_markers.append(float(row[6]))
    # print(x)
    # print(y)
    plt.figure()
    plt.plot(x, y, linewidth=1, color="green")
    # plt.plot(x_markers, y_markers, marker="o", markersize=5, markeredgecolor="blue")
    plt.xlabel("epoch")
    plt.ylabel("mAP@50")
    plt.savefig(args.out_file)
    print(f"Done. Figure is saved at {args.out_file}")

