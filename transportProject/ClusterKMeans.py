import numpy as np
import time

import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt


def readCSV(inputFile):
    my_matrix = np.loadtxt(open(inputFile, "rb"), delimiter=",", skiprows=0)
    return my_matrix


def writeCSV(outerMatrix, outputFile):
    np.savetxt(outputFile, outerMatrix, delimiter=',')


def cluster(V, clusterNum, whichone=False):
    # print(V)
    print("reading finish")

    if whichone:
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=clusterNum, batch_size=500,
                              n_init=20, max_no_improvement=10, verbose=0)
        t0 = time.time()
        mbk.fit(V)
        t_mini_batch = time.time() - t0
    else:
        mbk = KMeans(init='k-means++', n_clusters=clusterNum, n_jobs=-1,
                     n_init=20, verbose=0)
        t0 = time.time()
        mbk.fit(V)
        t_mini_batch = time.time() - t0
    print("the time: " + bytes(t_mini_batch))
    labels = np.array(mbk.labels_).T
    outerMatrix = np.column_stack((labels, V))
    return outerMatrix


def estimateKMeans(V, clusterStart=100, clusterEnd=300, dividedSpace=10, whichone=False):
    clusterNum = []
    clusterDistance = []
    if whichone:
        print("KMeans estimate:")
        for i in range(clusterStart, clusterEnd + 1, dividedSpace):
            mbk = KMeans(init='k-means++', n_clusters=i,
                         n_init=20, verbose=0)
            t0 = time.time()
            mbk.fit(V)
            t_mini_batch = time.time() - t0
            print(bytes(i) + "th time: " + bytes(t_mini_batch))
            clusterNumAverage = int(np.average(pd.Series(mbk.labels_).value_counts()))
            clusterInnerDistanceAvg = np.average(pd.Series(mbk.inertia_))
            clusterNum.append(clusterNumAverage)
            clusterDistance.append(clusterInnerDistanceAvg)
    else:
        print("MiniBatchKMeans estimate:")
        for i in range(clusterStart, clusterEnd + 1, dividedSpace):
            mbk = MiniBatchKMeans(init='k-means++', n_clusters=i, batch_size=500,
                                  n_init=20, max_no_improvement=10, verbose=0)
            t0 = time.time()
            mbk.fit(V)
            t_mini_batch = time.time() - t0
            print(bytes(i) + "th time: " + bytes(t_mini_batch))
            clusterNumAverage = int(np.average(pd.Series(mbk.labels_).value_counts()))
            clusterInnerDistanceAvg = np.average(pd.Series(mbk.inertia_))
            clusterNum.append(clusterNumAverage)
            clusterDistance.append(clusterInnerDistanceAvg)

    plt.plot(range(len(clusterNum)), clusterNum)
    plt.plot(range(len(clusterDistance)), clusterDistance)
    plt.show()


if __name__ == '__main__':
    # V1 = readCSV("C:\\Users\\11877\\Desktop\\20111121V.csv")[:,1:]
    # V2 = readCSV("C:\\Users\\11877\\Desktop\\20111122V.csv")[:,1:]
    # V = np.around((V1 + V2) / 2.0)
    # writeCSV(V,"C:\\Users\\11877\\Desktop\\weekdaysV.csv")
    TEMP1 = readCSV("C:\\Users\\11877\\Desktop\\temp\\20111122V.csv")[:, 1:]
    # TEMP2 = readCSV("C:\\Users\\11877\\Desktop\\temp\\20111121V.csv")[:, 1:]
    # TEMP = (TEMP1+TEMP2) / 2.0
    V = cluster(TEMP1, 220,whichone=True)
    writeCSV(V,"C:\\Users\\11877\\Desktop\\temp\\weekdaysV.csv")
    # V = readCSV("C:\\Users\\11877\\Desktop\\20111120W.csv")
    # estimateKMeans(V)
