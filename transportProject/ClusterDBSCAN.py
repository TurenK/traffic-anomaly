# -*- coding: UTF-8 -*-
import numpy as np
import time
from numpy import math
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

UNCLASSIFIED = False
NOISE = -1


def readCSV():
    my_matrix = np.loadtxt(open("C:\\Users\\11877\\Desktop\\20111120W.csv", "rb"), delimiter=",", skiprows=0)
    return my_matrix


def writeCSV(outerMatrix, outputFile):
    np.savetxt(outputFile, outerMatrix, delimiter=',')


# 在pointId点eps范围内的点的id集合
def region_query(data, pointId, eps):
    nPoints = data.shape[0]
    seeds = []
    for i in range(nPoints):
        if math.sqrt(np.power(data[pointId, :] - data[i, :], 2).sum()) < eps:
            seeds.append(i)
    return seeds


# 判断是否是噪声点
def isNoise(data, clusterResult, pointId, clusterId, eps, minPts):
    seeds = region_query(data, pointId, eps)  # 获取在eps范围内的点的id

    # 不满足minPts条件的为噪声点
    if len(seeds) < minPts:
        clusterResult[pointId] = NOISE
        return True
    return False


# 分类是否成功, clusterResult为分类结果
def clusterExpand(data, clusterResult, pointId, clusterId, eps, minPts):
    seeds = region_query(data, pointId, eps)  # 获取在eps范围内的点的id

    # 递归到非核心点
    if len(seeds) < minPts:
        clusterResult[pointId] = clusterId
        return True

    # 递归到下一个核心点
    else:
        clusterResult[pointId] = clusterId
        for seedId in seeds:
            clusterExpand(data, clusterResult, seedId, clusterId, eps, minPts)


# 计算某个类内部距离


# 计算类之间距离

# 得到分类结果和分类个数
def dbscan(data, eps, minPts):
    clusterId = 1
    nPoints = data.shape[0]
    clusterResult = [UNCLASSIFIED] * nPoints
    for pointId in range(nPoints):
        if clusterResult[pointId] == UNCLASSIFIED:
            if isNoise(data, clusterResult, pointId, clusterId, eps, minPts):
                continue
            if clusterExpand(data, clusterResult, pointId, clusterId, eps, minPts):
                clusterId = clusterId + 1

    return clusterResult, clusterId - 1


# 绘制柱形图
def plotPla(clusterResult, clusterId):
    subcluster = []
    for i in clusterResult:
        if clusterResult[i] == UNCLASSIFIED:
            subcluster[0] = subcluster[0] + 1
        elif clusterResult[i] == NOISE:
            subcluster[1] = subcluster[1] + 1
        else:
            subcluster[clusterResult[i]] = subcluster[clusterResult[i]] + 1

    n_groups = clusterId
    means_EveryGroup = subcluster

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.05

    opacity = 0.4
    rects1 = plt.bar(index, means_EveryGroup, bar_width, alpha=opacity, color='b', label='Men')

    plt.xlabel('Cluster')
    plt.ylabel('Num')
    plt.title('Clusters & Num of Every Group')
    xname = []
    for ii in range(clusterId):
        xname[ii] = "C" + ii
    plt.xticks(index + bar_width, xname)
    plt.ylim(0, 0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    V = np.array(readCSV())
    mbk = DBSCAN(eps=0.00000005,min_samples=3)
    t0 = time.time()
    mbk.fit(V)
    t_mini_batch = time.time() - t0
    labels = mbk.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print "Estimated number of clusters: %d" % n_clusters_
    ratio = (len(labels[labels == -1])*1.0) / (len(labels) *1.0)
    print("ratio: ", format(ratio,".3%"))
    labels = np.array(labels.T)
    outerMatrix = np.column_stack((labels, V))
    writeCSV(outerMatrix, "C:\\Users\\11877\\Desktop\\20111120WDB.csv")

    # (clusterResult, clusterId) = dbscan(V, 0.45, 10)
    # plotPla(clusterResult, clusterId)


# 分类是否成功, clusterResult为分类结果
def expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
    seeds = region_query(data, pointId, eps)  # 获取在eps范围内的点的id

    # 不满足minPts条件的为噪声点
    if len(seeds) < minPts:
        clusterResult[pointId] = NOISE
        return False

    # 满足条件, 作为某个类的起始点开始扩张该类
    else:
        clusterResult[pointId] = clusterId  # 划分到该簇
        for seedId in seeds:
            clusterResult[seedId] = clusterId

        while len(seeds) > 0:  # 持续扩张
            currentPoint = seeds[0]
            queryResults = region_query(data, currentPoint, eps)
            if len(queryResults) >= minPts:
                for i in range(len(queryResults)):
                    resultPoint = queryResults[i]
                    if clusterResult[resultPoint] == UNCLASSIFIED:
                        seeds.append(resultPoint)
                        clusterResult[resultPoint] = clusterId
                    elif clusterResult[resultPoint] == NOISE:
                        clusterResult[resultPoint] = clusterId
            seeds = seeds[1:]
        return True


def plotFeature(data, clusters, clusterNum):
    nPoints = data.shape[1]
    matClusters = np.mat(clusters).transpose()
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)
    for i in range(clusterNum + 1):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[:, np.nonzero(matClusters[:, 0].A == i)]
        ax.scatter(subCluster[0, :].flatten().A[0], subCluster[1, :].flatten().A[0], c=colorSytle, s=50)
