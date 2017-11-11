from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sklearn
import numpy as np


def get_labels(X,k):
    new_x = PCA(n_components=0.9).fit_transform(X)
    k_means_method = KMeans(n_clusters=k)
    k_means_method.fit(new_x)
    return k_means_method.labels_, k_means_method.inertia_

