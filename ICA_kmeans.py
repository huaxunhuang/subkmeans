from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans


def get_labels(X,k):
    new_x = FastICA(n_components=k).fit_transform(X)
    k_means_method = KMeans(n_clusters=k)
    k_means_method.fit(new_x)
    return k_means_method.labels_, k_means_method.inertia_


