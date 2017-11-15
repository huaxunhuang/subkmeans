from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as ppt
import pandas as pd
import scipy.io as sio
from pandas.plotting import scatter_matrix
import importlib
delta = 1e-6

methods = ["LDA_kmeans","PCA_kmeans","ICA_kmeans"]
def dataMatOutput(features, labels, parameters):
    for key in features.keys():
        feature, label = features[key].as_matrix(), labels[key].as_matrix()
        feature = preprocessing.scale(feature)
        new_label = np.ones_like(label).astype("double")
        for idx,value in enumerate(np.unique(label)):
            new_label[label==value]=idx
        sio.savemat("data/mat/"+key+".mat",{"feature":feature,"label":new_label})

def dataOutput(features, labels, parameters):
    for key in features.keys():
        f = open("data/standard/"+key,"w")
        feature, label = features[key].as_matrix(), labels[key].as_matrix()
        feature = preprocessing.scale(feature)
        for i in range(parameters[key][0]):
            for j in range(parameters[key][1]):
                f.write(str(feature[i,j])+",")
            f.write("C"+str(label[i])+"\n")
        f.close()

def dataCollection():
    #data collection
    features = dict()
    labels = dict()
    parameters = dict()  # sample number, dimension, k
    #wine. paper use 9 dimensions, while there are 13 dimensions.
    dataset = pd.read_csv("data/wine.data", header=None)
    features["wine"] = dataset.loc[:,1:13]
    labels["wine"] = dataset.loc[:,0]
    np.unique(labels["wine"])
    parameters["wine"] = [dataset.shape[0],dataset.shape[1]-1,len(np.unique(labels["wine"]))]

    #pendigit: paper use training data.
    dataset = pd.read_csv("data/pendigits.tra", header=None, sep=",")
    features["pendigits"] = dataset.loc[:,:dataset.shape[1]-2]
    labels["pendigits"] = dataset.loc[:,dataset.shape[1]-1]
    parameters["pendigits"] = [dataset.shape[0],dataset.shape[1]-1,len(np.unique(labels["pendigits"]))]

    #Ecoli
    dataset = pd.read_csv("data/ecoli.data", header=None,delim_whitespace=True)
    features["ecoli"] = dataset.loc[:,1:dataset.shape[1]-2]
    labels["ecoli"] = dataset.loc[:,dataset.shape[1]-1]
    parameters["ecoli"] = [dataset.shape[0],dataset.shape[1]-2,len(np.unique(labels["ecoli"]))]

    #Seeds
    dataset = pd.read_csv("data/seeds_dataset.txt", header=None,delim_whitespace=True)
    features["seeds"] = dataset.loc[:,:dataset.shape[1]-2]
    labels["seeds"] = dataset.loc[:,dataset.shape[1]-1]
    parameters["seeds"] = [dataset.shape[0],dataset.shape[1]-1,len(np.unique(labels["seeds"]))]

    #Soybean
    dataset = pd.read_csv("data/soybean-small.data", header=None)
    features["soybean"] = dataset.loc[:,:dataset.shape[1]-2]
    labels["soybean"] = dataset.loc[:,dataset.shape[1]-1]
    parameters["soybean"] = [dataset.shape[0],dataset.shape[1]-1,len(np.unique(labels["soybean"]))]

    #Symbol
    dataset = pd.read_csv("data/Symbols_TEST.arff", header=None)
    features["symbol"] = dataset.loc[:,:dataset.shape[1]-2]
    labels["symbol"] = dataset.loc[:,dataset.shape[1]-1]
    parameters["symbol"] = [dataset.shape[0],dataset.shape[1]-1,len(np.unique(labels["symbol"]))]

    #OliveOil
    dataset = pd.read_csv("data/OliveOil.arff", header=None)
    features["oliveoil"] = dataset.loc[:,:dataset.shape[1]-2]
    labels["oliveoil"] = dataset.loc[:,dataset.shape[1]-1]
    parameters["oliveoil"] = [dataset.shape[0],dataset.shape[1]-1,len(np.unique(labels["oliveoil"]))]

    #Plane
    dataset = pd.read_csv("data/Plane.arff", header=None)
    features["plane"] = dataset.loc[:,:dataset.shape[1]-2]
    labels["plane"] = dataset.loc[:,dataset.shape[1]-1]
    parameters["plane"] = [dataset.shape[0],dataset.shape[1]-1,len(np.unique(labels["plane"]))]

    return features, labels, parameters


def algorithmEval(method, feature, label, k, iter=40):
    cost, NMI = np.zeros(iter),np.zeros(iter)
    feature = preprocessing.scale(feature)
    for i in range(iter):
        cluster_result, cost[i] = method(feature, k)
        NMI[i] = normalized_mutual_info_score(cluster_result, label)
    return np.mean(NMI[np.argsort(cost)][:(1+iter)//2])


if __name__ =="__main__":
    features, labels, parameters = dataCollection()
    dataOutput(features,labels,parameters)
    dataMatOutput(features,labels,parameters)
    for dataset_name in features.keys():
        print("DATASET: %s K: %d SAMPLE NUMBER: %d DIMENSION: %d"%(dataset_name, parameters[dataset_name][2],
                                                                   parameters[dataset_name][0],
                                                                   parameters[dataset_name][1]))
    for method_name in methods:
        m_module = importlib.import_module(method_name)
        for dataset_name in features.keys():
            NMI = algorithmEval(m_module.get_labels, features[dataset_name], labels[dataset_name], parameters[dataset_name][2])
            print("METHOD:%s\tDATASET:%s\t%f"%(method_name, dataset_name, NMI))
            if parameters[dataset_name][1] < 10:
                pass
                # this is for labels... useless for clusters.
                # c = clusters[dataset_name].copy()
                # for i,group in enumerate(np.unique(clusters[dataset_name])):
                #    c[c==group] = i
                # print(dataset_name+"-DRAW!")
                # if dataset_name=="wine":
                #    _ = scatter_matrix(dataset[dataset_name].loc[:,1:features[dataset_name][1]],marker="o",c=clusters[dataset_name])
                # elif dataset_name=="ecoli":
                #    _ = scatter_matrix(dataset[dataset_name].loc[:,1:features[dataset_name][1]],marker="o",c=clusters[dataset_name])
                # else:
                #    _ = scatter_matrix(dataset[dataset_name].loc[:,:features[dataset_name][1]-1],marker="o",c=clusters[dataset_name])
