import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Draw report 1', add_help=False)
    # parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('npy_file_1', type=str) # all_feature_3.npy / all_feature_1.npy
    parser.add_argument('npy_file_2', type=str) # all_feature_4.npy / all_feature_2.npy
    parser.add_argument('title', type=str) # 't-SNE: Adapted model inference'
    parser.add_argument('out_png_name', type=str) # "./report_bonus34.png"
    args = parser.parse_args()

    features_A = np.load(args.npy_file_1)
    labels_A = np.zeros(features_A.shape[0])

    features_B = np.load(args.npy_file_2)
    labels_B = np.ones(features_B.shape[0])

    features = np.concatenate((features_A, features_B), axis=0)
    groups = np.concatenate((labels_A, labels_B), axis=0)
    print(features.shape)
    print(groups.shape)

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    features_tsne = tsne.fit_transform(features) 


    classes=["clear-val", "foggy-val"]
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], s=10, c=groups, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    n_groups = len(classes)
    cb1 = plt.colorbar(boundaries=np.arange(n_groups+1)-0.5).set_ticks(np.arange(n_groups))
    plt.title(args.title, fontsize=12)
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    save_img_as = args.out_png_name # = os.path.join(output_dir, f"Report 2-3 usps tsne by class.png")
    plt.savefig(save_img_as)
