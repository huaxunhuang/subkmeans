import sklearn
import numpy as np

from sklearn.cluster import KMeans

def get_labels(X):
    k_means_method = KMeans(n_clusters=3)
    k_means_method.fit(X)
    return k_means_method.labels_