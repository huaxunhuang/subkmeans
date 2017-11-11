from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as ppt
import pandas as pd
from pandas.plotting import scatter_matrix
import importlib
delta = 1e-6

methods = ["kmeans","PCA_kmeans","ICA_kmeans"]
clusters = dict()
labels = dict()
features = dict()#sample number, dimension, k
dataset = dict()

def getAverageResult(data,method,avgSum,k):
    data = preprocessing.scale(data)
    return method(data,k)[0]

for method in methods:
    m_module = importlib.import_module(method)
    print()
    print("METHOD: "+method)
    k=3
    #wine. paper use 9 dimensions, while there are 13 dimensions.
    dataset["wine"] = pd.read_csv("data/wine.data", header=None)
    x = dataset["wine"].loc[:,1:13]
    clusters["wine"] = getAverageResult(x,m_module.get_labels,avgSum=40,k=k)
    labels["wine"] = dataset["wine"].loc[:,0]
    features["wine"] = [dataset["wine"].shape[0],dataset["wine"].shape[1]-1,k]

    #pendigit: paper use training data.
    k=10
    dataset["pendigits"] = pd.read_csv("data/pendigits.tra", header=None, sep=",")
    x = dataset["pendigits"].loc[:,:dataset["pendigits"].shape[1]-2]
    clusters["pendigits"] = getAverageResult(x,m_module.get_labels,avgSum=40,k=k)
    labels["pendigits"] = dataset["pendigits"].loc[:,dataset["pendigits"].shape[1]-1]
    features["pendigits"] = [dataset["pendigits"].shape[0],dataset["pendigits"].shape[1]-1,k]

    #Ecoli
    k=5
    dataset["ecoli"] = pd.read_csv("data/ecoli.data", header=None,delim_whitespace=True)
    x = dataset["ecoli"].loc[:,1:dataset["ecoli"].shape[1]-2]
    clusters["ecoli"] = getAverageResult(x,m_module.get_labels,avgSum=40,k=k)
    labels["ecoli"] = dataset["ecoli"].loc[:,dataset["ecoli"].shape[1]-1]
    features["ecoli"] = [dataset["ecoli"].shape[0],dataset["ecoli"].shape[1]-2,k]


    #Seeds
    k=3
    dataset["seeds"] = pd.read_csv("data/seeds_dataset.txt", header=None,delim_whitespace=True)
    x = dataset["seeds"].loc[:,:dataset["seeds"].shape[1]-2]
    clusters["seeds"] = getAverageResult(x,m_module.get_labels,avgSum=40,k=k)
    labels["seeds"] = dataset["seeds"].loc[:,dataset["seeds"].shape[1]-1]
    features["seeds"] = [dataset["seeds"].shape[0],dataset["seeds"].shape[1]-1,k]


    #Soybean
    k=4
    dataset["soybean"] = pd.read_csv("data/soybean-small.data", header=None)
    x = dataset["soybean"].loc[:,:dataset["soybean"].shape[1]-2]
    clusters["soybean"] = getAverageResult(x,m_module.get_labels,avgSum=40,k=k)
    labels["soybean"] = dataset["soybean"].loc[:,dataset["soybean"].shape[1]-1]
    features["soybean"] = [dataset["soybean"].shape[0],dataset["soybean"].shape[1]-1,k]


    #Symbol
    k=6
    dataset["symbol"] = pd.read_csv("data/Symbols_TEST.arff", header=None)
    x = dataset["symbol"].loc[:,:dataset["symbol"].shape[1]-2]
    clusters["symbol"] = getAverageResult(x,m_module.get_labels,avgSum=40,k=k)
    labels["symbol"] = dataset["symbol"].loc[:,dataset["symbol"].shape[1]-1]
    features["symbol"] = [dataset["symbol"].shape[0],dataset["symbol"].shape[1]-1,k]


    #OliveOil
    k=4
    dataset["oliveoil"] = pd.read_csv("data/OliveOil.arff", header=None)
    x = dataset["oliveoil"].loc[:,:dataset["oliveoil"].shape[1]-2]
    clusters["oliveoil"] = getAverageResult(x,m_module.get_labels,avgSum=40,k=k)
    labels["oliveoil"] = dataset["oliveoil"].loc[:,dataset["oliveoil"].shape[1]-1]
    features["oliveoil"] = [dataset["oliveoil"].shape[0],dataset["oliveoil"].shape[1]-1,k]


    #Plane
    k=7
    dataset["plane"] = pd.read_csv("data/Plane.arff", header=None)
    x = dataset["plane"].loc[:,:dataset["plane"].shape[1]-2]
    clusters["plane"] = getAverageResult(x,m_module.get_labels,avgSum=40,k=k)
    labels["plane"] = dataset["plane"].loc[:,dataset["plane"].shape[1]-1]
    features["plane"] = [dataset["plane"].shape[0],dataset["plane"].shape[1]-1,k]


    for dataset_name in labels.keys():
        print(dataset_name+"-NMI:",normalized_mutual_info_score(clusters[dataset_name], labels[dataset_name]))
        #print(features[dataset_name])
        if features[dataset_name][1]<10:
            pass
            #this is for labels... useless for clusters.
            #c = clusters[dataset_name].copy()
            #for i,group in enumerate(np.unique(clusters[dataset_name])):
            #    c[c==group] = i
            #print(dataset_name+"-DRAW!")
            #if dataset_name=="wine":
            #    _ = scatter_matrix(dataset[dataset_name].loc[:,1:features[dataset_name][1]],marker="o",c=clusters[dataset_name])
            #elif dataset_name=="ecoli":
            #    _ = scatter_matrix(dataset[dataset_name].loc[:,1:features[dataset_name][1]],marker="o",c=clusters[dataset_name])
            #else:
            #    _ = scatter_matrix(dataset[dataset_name].loc[:,:features[dataset_name][1]-1],marker="o",c=clusters[dataset_name])
