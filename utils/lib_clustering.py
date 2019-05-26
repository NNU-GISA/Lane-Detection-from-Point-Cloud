'''
Clustering by DBSCAN using sklearn library

This code is copied and modified from:
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
'''

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class Clusterer(object):

    def __init__(self):
        self.fit_success = False

    def fit(self, X, eps=0.3, min_samples=10):
        # Compute DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        # samples that is close to the center
        core_samples_mask[db.core_sample_indices_] = True
        self.X = X
        self.db = db
        self.core_samples_mask = core_samples_mask
        self.fit_success = True
        self.labels = db.labels_  # label of each sample
        self.unique_labels = set(self.labels)
        self.n_clusters = len(set(self.labels)) - \
            (1 if -1 in self.labels else 0)

    def plot_clusters(self):
        if not self.fit_success:
            return
        assert self.X.shape[1] == 2, "To visualize result, X must be 2 dimenstions."

        # member vars used in this function
        labels, n_clusters, unique_labels = self.labels, self.n_clusters, self.unique_labels
        core_samples_mask = self.core_samples_mask
        X = self.X

        # Black removed and is used for noise instead.
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        # print(colors)
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)
            # break
        plt.title('Clustering result: {} clusters'.format(n_clusters))

    def print_clustering_result(self):
        if not self.fit_success:
            return
        labels, n_clusters = self.labels, self.n_clusters

        # Number of clusters in labels, ignoring noise if present.
        n_noise_ = list(labels).count(-1)

        print('Estimated number of clusters: %d' % n_clusters)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Homogeneity: %0.3f" %
              metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" %
              metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" %
              metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f"
              % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f"
              % metrics.adjusted_mutual_info_score(labels_true, labels,
                                                   average_method='arithmetic'))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(X, labels))


if __name__ == "__main__":
    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                random_state=0)
    X = StandardScaler().fit_transform(X)

    # Fit
    cluster = Clusterer()
    cluster.fit(X)

    # Plot
    cluster.plot_clusters()
    plt.show()
