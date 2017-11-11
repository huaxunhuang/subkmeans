import sklearn
import numpy as np

from sklearn.cluster import KMeans

def get_labels(X,k):
    k_means_method = KMeans(n_clusters=k)
    k_means_method.fit(X)
    return k_means_method.labels_, k_means_method.inertia_