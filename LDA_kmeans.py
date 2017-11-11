from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import math

def get_labels(X,k):
    max_iteration = 100
    lastCost, cost = 0,0
    k_means_method = KMeans(n_clusters=k)
    clf = LinearDiscriminantAnalysis(n_components=k - 1)
    X = PCA(n_components=k-1).fit_transform(X)
    for i in range(max_iteration):
        k_means_method.fit(X)
        label, cost = k_means_method.labels_, k_means_method.inertia_
        if math.fabs(cost-lastCost) < 1e-1:
            break
        lastCost = cost
        X = clf.fit_transform(X, label)
        #print("%d:%f"%(i,cost-lastCost))
    return label, cost