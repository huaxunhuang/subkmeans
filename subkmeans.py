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

    miu_d = []
    # Line 7
    for arr in dataset:
        arr.shape = (shape(arr)[0],1)
        if miu_d == []:
            miu_d = arr

        miu_d = miu_d + np.array(arr)
    miu_d = miu_d * (1.0 / len(dataset))

    # Line 8
    s_d = []
    for arr in dataset:
        arr.shape = (shape(arr)[0],1)
        temp = arr - miu_d
        temp.shape = (shape(temp)[0],1)

        res = temp.dot(transpose(temp))
        if s_d == []:
            s_d = res
        else:
            s_d = s_d+res

    # Line 9
    miu_cluster = random.sample(dataset, k)
    # randomly select k numbers

    cost_value = sys.maxint
    # # Line 11
    while True:
        # P_c and P_n
        P_c = eye(m)
        P_c = np.vstack((P_c, zeros([d - m, m])))
        P_n = zeros([m, d - m])
        P_n = np.vstack((P_n, eye(d - m)))

        labels = []
        # assignment step
        clusters = []
        for i in range(0, k):
            clusters.append([])

        # Line 12
        for arr in dataset:
            # Line 13
            j = -1
            jvalue = sys.maxint
            for i in range(0, k):
                tempvalue = P_c.T.dot(V.T).dot(arr)
                tempvalue = tempvalue - P_c.T.dot(V.T).dot(miu_cluster[i])
                dist = np.linalg.norm(tempvalue)
                dist = dist * dist
                if dist < jvalue:
                    jvalue = dist
                    j = i
            clusters[j].append(np.array(arr))
            labels.append(j)

        # update step
        S_clusters = []
        for i in range(0, k):
            S_clusters.append([])

        for i in range(0, k):
            C_i = shape(clusters[i])[0]
            sum = []
            for arr in clusters[i]:
                if sum == []:
                    sum = arr
                else:
                    sum = sum + arr
            # Line 16
            miu_cluster[i] = sum * (1.0 / C_i)

            sum = []
            # Line 17
            for arr in clusters[i]:
                temp1 = np.matrix(arr - miu_cluster[i])
                if sum == []:
                    sum = temp1.T.dot(temp1)
                else:
                    sum = sum + temp1.T.dot(temp1)
            S_clusters[i] = sum

        # Line 18
        sum = []
        for i in range(0, k):
            # print shape(S_clusters[i])
            if sum == []:
                sum = S_clusters[i]
            else:
                sum = sum + S_clusters[i]
        sum = sum - s_d
        e, V = np.linalg.eig(sum)
        # order e and V
        etemp = np.array(sorted(e))
        e = etemp
        V = np.array(V)
        Vtemp = []
        for num in range(0,shape(etemp)[0]):
            Vtemp.append(V.T[np.where(e == etemp[num])[0][0]])
        Vtemp = np.array(Vtemp).T
        V = Vtemp

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
                tempval = P_c.T.dot(V.T).dot(arr)-P_c.T.dot(V.T).dot(miu_cluster[i])
                tempval = np.linalg.norm(tempval)
                tempval = tempval * tempval
                cost_value_temp = cost_value_temp + tempval
                tempval = P_n.T.dot(V.T).dot(arr)-P_n.T.dot(V.T).dot(miu_d)
                tempval = np.linalg.norm(tempval)
                tempval = tempval * tempval
                cost_value_temp = cost_value_temp+tempval

        if cost_value_temp < cost_value:
            cost_value = cost_value_temp
        else:
            break

        dimensions = []
        for j in range(0, shape(e)[0]):
            if e[j] < 0:
                dimensions.append(j)
                dimensions.append(e[j])

    print "m:"+str(m)
    return labels,cost_value
