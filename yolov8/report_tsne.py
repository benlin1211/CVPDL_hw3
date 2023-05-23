import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

features_A = np.load("all_feature_1.npy")
labels_A = np.zeros(features_A.shape[0])

features_B = np.load("all_feature_2.npy")
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
plt.title('t-SNE: Source model inference', fontsize=12)
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
save_img_as = "./report_bonus12.png" # = os.path.join(output_dir, f"Report 2-3 usps tsne by class.png")
plt.savefig(save_img_as)
