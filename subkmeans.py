import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import random
import sys


def get_labels(dataset, k):
    labels = []
    dataset = np.array(dataset)
    d = dataset.shape[1]

    # Initialization
    randomMat = np.random.randn(d, d)
    q, r = np.linalg.qr(randomMat)
    V = q
    m = d / 2

    # P_c and P_n
    P_c = eye(m)
    P_c = np.vstack((P_c, zeros([d - m, m])))
    P_n = zeros([m, d - m])
    P_n = np.vstack((P_n, eye(d - m)))

    # Line 7
    miu_d = zeros([1, d])
    for arr in dataset:
        miu_d = miu_d + arr

    miu_d = miu_d * (1.0 / len(dataset))

    # Line 8
    s_d = zeros([d, d])
    for arr in dataset:
        arr1 = np.array(arr).T
        temp = miu_d - arr1
        res = temp.T.dot(temp)
        s_d = s_d + res

    # Line 9
    miu_cluster = random.sample(dataset, k);
    # randomly select k numbers

    clusters = []
    for i in range(0, k):
        clusters.append([])

    cost_value = sys.maxint
    # Line 11
    while True:
        labels = []
        # assignment step
        clusters = []
        for i in range(0, k):
            clusters.append([])

            # Line 12
        #         print shape(dataset)

        for arr in dataset:
            # Line 13
            j = -1
            jvalue = sys.maxint
            for i in range(0, k):
                tempvalue = P_c.T.dot(V.T).dot(arr)
                tempvalue = tempvalue - P_c.T.dot(V.T).dot(miu_cluster[i])
                dist = np.linalg.norm(tempvalue)
                if dist < jvalue:
                    jvalue = dist
                    j = i

            clusters[j].append(arr)
            labels.append(j)

        # update step
        S_clusters = [];
        for i in range(0, k):
            S_clusters.append([])

        for i in range(0, k):
            C_i = shape(clusters[i])[0]
            sum = zeros(shape(clusters[i][0]))
            for arr in clusters[i]:
                sum = sum + arr
            # Line 16
            miu_cluster[i] = sum * (1.0 / C_i)

            sum = zeros([d, d])
            # Line 17
            for arr in clusters[i]:
                temp1 = np.matrix(arr - miu_cluster[i])
                sum = sum + temp1.T.dot(temp1)

            S_clusters[i] = sum
            # print miu_cluster[i]

        # Line 18
        sum = zeros(shape(S_clusters[0]))
        for i in range(0, k):
            sum = sum + S_clusters[i]

        sum = sum - s_d
        e, v = np.linalg.eig(sum)
        #         print "e:"+str(e)

        # Line 19
        tempm = 0
        for val in e:
            if val < 0:
                tempm = tempm + 1
        m = tempm
        #         print "m:"+str(m)

        # convergence
        cost_value_temp = 0.0
        for i in range(0, k):
            for arr in clusters[i]:
                tempval = np.dot(np.dot(P_c.T,V.T),arr)-np.dot(np.dot(P_c.T,V.T),miu_cluster[i])
                tempval = np.linalg.norm(tempval)
                tempval = tempval * tempval
                cost_value_temp = cost_value_temp + tempval
                tempval = np.dot(np.dot(P_n.T,V.T),arr)-np.dot(np.dot(P_n.T,V.T),miu_d.T)
                tempval = np.linalg.norm(tempval)
                tempval = tempval * tempval
                cost_value_temp = cost_value_temp+tempval

        if cost_value_temp<cost_value:
            print cost_value
            cost_value = cost_value_temp
        else:
            break

        dimensions = []
        for j in range(0, shape(e)[0]):
            if e[j] < 0:
                dimensions.append(j)
                dimensions.append(e[j])

    return labels,cost_value
