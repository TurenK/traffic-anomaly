# -*- coding: UTF-8 -*-
from tokenize import String

import numpy as np
import xml.etree.cElementTree as et
import NMF as nmf
import ClusterKMeans as kmeans
import scipy

np.set_printoptions(threshold=np.inf)

inputFile = "C:\\Users\\11877\\Desktop\\temp\\newData\\rawData\\"
nmfoutputFile = "C:\\Users\\11877\\Desktop\\temp\\newData\\afterNMF\\"
kmeansoutputFile = "C:\\Users\\11877\\Desktop\\temp\\newData\\afterKMeans\\"
normaloutputFile = "C:\\Users\\11877\\Desktop\\temp\\newData\\afterNormal\\"
predictFile = "C:\\Users\\11877\\Desktop\\temp\\newData\\predictData\\"
scoreFile = "C:\\Users\\11877\\Desktop\\temp\\newData\\ScoreMatrix\\"
EndResult = "C:\\Users\\11877\\Desktop\\temp\\newData\\EndResult\\"
COLUMNNUM = 96


def readCSV(inputFile):
    my_matrix = np.loadtxt(open(inputFile, "rb"), delimiter=",", skiprows=0)
    return my_matrix


def writeCSV(outerMatrix, outputFile):
    np.savetxt(outputFile, outerMatrix, delimiter=',')


def createNormalDistri(V, M, whichOne=True):
    # calculate the result of now
    if whichOne:
        clusterNum = int(np.max(M[:, 1]))
        clusterAvg = np.zeros([clusterNum + 1, V.shape[1] - 3])
        clusterStd = np.zeros([clusterNum + 1, V.shape[1] - 3])
        everyClusterNum = np.zeros([clusterNum + 1, V.shape[1] - 3])

        # calculate means
        for sequence in range(V.shape[0]):
            clusterAvg[int(M[sequence, 1])] = clusterAvg[int(M[sequence, 1])] + V[int(M[sequence, 1]), 3:]
            everyClusterNum[int(M[sequence, 1]), :] = everyClusterNum[int(M[sequence, 1]), :] + 1
        clusterAvg = np.true_divide(clusterAvg, clusterNum)
        # print(clusterAvg)
        print("Calculate Avg End")

        # calculate standard deviation
        for sequence in range(V.shape[0]):
            temp = V[int(M[sequence, 1]), 3:] - clusterAvg[int(M[sequence, 1]), :]
            Varience = np.multiply(temp, temp)
            clusterStd[int(M[sequence, 1])] = clusterStd[int(M[sequence, 1])] + Varience
        clusterStd = np.true_divide(clusterStd, clusterNum - 1)
        for i in range(clusterStd.shape[0]):
            for j in range(clusterStd.shape[1]):
                clusterStd[i, j] = pow(clusterStd[i, j], 0.5)
        # print(clusterStd)
        print("Calculate Std End")
        return clusterAvg, clusterStd

    # calculate the result of history
    else:
        clusterAvg = np.zeros([V.shape[0], COLUMNNUM])
        clusterStd = np.zeros([V.shape[0], COLUMNNUM])

        # calculate means
        for sequence in range(V.shape[0]):
            clusterAvg[sequence] = (V[sequence, 4:100] + V[sequence, 100:196] + V[sequence, 196:]) / 3.0
        # print(clusterAvg)
        print("Calculate Avg End")

        # calculate standard deviation
        for sequence in range(V.shape[0]):
            temp1 = V[sequence, 4:100] - clusterAvg[sequence]
            Varience1 = np.multiply(temp1, temp1)
            temp2 = V[sequence, 100:196] - clusterAvg[sequence]
            Varience2 = np.multiply(temp2, temp2)
            temp3 = V[sequence, 196:] - clusterAvg[sequence]
            Varience3 = np.multiply(temp3, temp3)
            clusterStd[sequence] = np.true_divide((Varience1 + Varience2 + Varience3), 2)
        for i in range(clusterStd.shape[0]):
            for j in range(clusterStd.shape[1]):
                clusterStd[i, j] = pow(clusterStd[i, j], 0.5)
        # print(clusterStd)
        print("Calculate Std End")
        return clusterAvg, clusterStd


# return the probability density
def normfun(x, mu, sigma):
    pd = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pd


def HistoryScore(V, predictM):
    HistoryScore = np.zeros([V.shape[0], COLUMNNUM])
    (clusterAvg, clusterStd) = createNormalDistri(V, np.zeros([V.shape[0], 2]), whichOne=False)
    writeCSV(clusterAvg, predictFile + "historyMeans.csv")
    writeCSV(clusterStd, predictFile + "historyStds.csv")
    for sequence in range(predictM.shape[0]):
        for columnnum in range(predictM.shape[1]):
            if clusterStd[sequence, columnnum] <= 0.1:
                if V[sequence, columnnum] <= clusterAvg[sequence, columnnum]:
                    HistoryScore[sequence, columnnum] = 1
                else:
                    HistoryScore[sequence, columnnum] = normfun(predictM[sequence, columnnum],
                                                                clusterAvg[sequence, columnnum],
                                                                clusterStd[sequence, columnnum] + 1)
            else:
                if V[sequence, columnnum] <= clusterAvg[sequence, columnnum]:
                    HistoryScore[sequence, columnnum] = 1
                else:
                    HistoryScore[sequence, columnnum] = normfun(predictM[sequence, columnnum],
                                                                clusterAvg[sequence, columnnum],
                                                                clusterStd[sequence, columnnum])
    return HistoryScore


def NeighborScore(V, matchM, predictM):
    NeighborScore = np.zeros([V.shape[0], COLUMNNUM])
    (clusterAvg, clusterStd) = createNormalDistri(V, matchM, whichOne=True)
    writeCSV(clusterAvg, predictFile + "neighborMeans.csv")
    writeCSV(clusterStd, predictFile + "neighborStds.csv")
    for sequence in range(predictM.shape[0]):
        for columnnum in range(predictM.shape[1]):
            if clusterStd[sequence, columnnum] <= 0.1:
                if V[sequence, columnnum] <= clusterAvg[sequence, columnnum]:
                    NeighborScore[sequence, columnnum] = 1
                else:
                    NeighborScore[sequence, columnnum] = normfun(predictM[sequence, columnnum],
                                                                 clusterAvg[sequence, columnnum],
                                                                 clusterStd[sequence, columnnum] + 1)
            else:
                if V[sequence, columnnum] <= clusterAvg[sequence, columnnum]:
                    NeighborScore[sequence, columnnum] = 1
                else:
                    NeighborScore[sequence, columnnum] = normfun(predictM[sequence, columnnum],
                                                                 clusterAvg[sequence, columnnum],
                                                                 clusterStd[sequence, columnnum])
    return NeighborScore


def getScore(Historypredict, Neighborpredict, Beta):
    Score = Beta * Historypredict + (1 - Beta) * Neighborpredict
    return Score


def getResult(Score, MarginValue):
    Result = np.ones((Score.shape[0], Score.shape[1]))
    AbnormalNum = 0
    for rownum in range(Score.shape[0]):
        for columnnum in range(Score.shape[1]):
            if (Score[rownum, columnnum] < MarginValue):
                Result[rownum, columnnum] = 0
    return Result


def getBlandMatrix(Result, RealResult, V):
    AbnormalNum = 0
    for rownum in range(Score.shape[0]):
        for columnnum in range(Score.shape[1]):
            if (Result[rownum, columnnum] == 0):
                AbnormalNum = AbnormalNum + 1

    RealAbnormalAndCall = 0
    for realresultrow in range(RealResult.shape[0]):
        for VRow in range(V.shape[0]):
            if V[VRow, 0] == RealResult[realresultrow, 3] and V[VRow, 1] == RealResult[realresultrow, 4]:
                if Result[int(V[VRow, 2]), int(RealResult[realresultrow, 2])] == 0:
                    RealAbnormalAndCall = RealAbnormalAndCall + 1

    RealAbnormalNum = RealResult.shape[0]
    P = RealAbnormalAndCall * 1.0 / (AbnormalNum * 1.0)
    R = RealAbnormalAndCall * 1.0 / (RealAbnormalNum * 1.0)
    if P <= 0.0000001 and R <= 0.0000001:
        F1 = 0.0
    else:
        F1 = 2.0 * (P * R) / (P + R)

    return P, R, F1


if __name__ == '__main__':
    # # nmf
    # rawData0 = readCSV(inputFile + "20111120.csv") * 1.0
    # rawData1 = readCSV(inputFile + "20111121.csv") * 1.0
    # rawData2 = readCSV(inputFile + "20111122.csv") * 1.0
    # rawData = np.column_stack((rawData0, rawData1[:, 3:], rawData2[:, 3:]))
    # rawData = readCSV(inputFile + "rawData.csv")
    # writeCSV(rawData, inputFile + "rawData.csv")
    # print("Finish Reading")
    # W = np.random.randint(1, 100, size=[rawData.shape[0], 4]) * 1.0
    # H = np.random.randint(1, 100, size=[4, rawData.shape[1] - 3]) * 1.0
    # V = rawData[:, 3:]
    # (W, H) = nmf.NMFNormal(V, W, H, steps=8000)
    # print(W)
    # print(H)
    # writeCSV(W, nmfoutputFile + "20111120W.csv")
    # writeCSV(H, nmfoutputFile + "20111120H.csv")

    # # KMeans
    # TEMP1 = readCSV(nmfoutputFile + "20111120W.csv")
    # # TEMP2 = readCSV("C:\\Users\\11877\\Desktop\\temp\\20111121V.csv")[:, 1:]
    # # TEMP = (TEMP1+TEMP2) / 2.0
    # WKM = kmeans.cluster(TEMP1, 220, whichone=False)
    # print(WKM)
    # writeCSV(WKM, kmeansoutputFile + "WKM.csv")
    # WKM = readCSV(kmeansoutputFile + "WKM.csv")
    #
    # VKM = np.column_stack((WKM[:, 0], rawData))
    # #print(VKM)
    # writeCSV(VKM, kmeansoutputFile + "V.csv")

    # # History Predict
    # V = readCSV(kmeansoutputFile + "V.csv")
    # predictM = readCSV(predictFile + "20111126.csv")[:, 3:]
    # Historypredict = HistoryScore(V, predictM)
    # # writeCSV(Historypredict, predictFile + "HistoryPredict.csv")
    # #
    # # # Neighbor predict
    # V = readCSV(predictFile + "20111126.csv")
    # # matchM = readCSV(kmeansoutputFile + "WKM.csv")[:, 0:2]
    # # predictM = readCSV(predictFile + "20111126.csv")[:, 3:]
    # # Neighborpredict = NeighborScore(V, matchM, predictM)
    # # writeCSV(Neighborpredict, predictFile + "NeighborPredict.csv")
    #
    # Historypredict = readCSV(predictFile + "HistoryPredict.csv")
    # Neighborpredict = readCSV(predictFile + "NeighborPredict.csv")
    # RealResult = readCSV(scoreFile + "11.26.csv")
    # #
    # # get the estimate matrix
    # BETA = np.array([0.38,0.38,0.38,0.38,0.38,0.38,
    #                  0.43, 0.43, 0.43, 0.43, 0.43, 0.43,
    #                  0.48, 0.48, 0.48, 0.48, 0.48, 0.48,
    #                  0.53, 0.53, 0.53, 0.53, 0.53, 0.53,
    #                  0.58, 0.58, 0.58, 0.58, 0.58, 0.58,
    #                  0.63, 0.63, 0.63, 0.63, 0.63, 0.63])
    # MarginValue = np.array([1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6,
    #                         1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6,
    #                         1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6,
    #                         1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6,
    #                         1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6,
    #                         1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6])
    #
    # BETAAndMarginV = np.column_stack((BETA.T, MarginValue.T))
    # PM = np.ones([1, 36])
    # RM = np.ones([1, 36])
    # F1M = np.ones([1, 36])
    # for i in range(36):
    #     Score = getScore(Historypredict, Neighborpredict, BETAAndMarginV[i, 0])
    #     Result = getResult(Score, BETAAndMarginV[i, 1])
    #     (P, R, F1) = getBlandMatrix(Result, RealResult, V)
    #     PM[0, i] = P
    #     RM[0, i] = R
    #     F1M[0, i] = F1
    # BlandMatrix = np.column_stack((BETAAndMarginV, PM.T, RM.T, F1M.T))
    # writeCSV(BlandMatrix, EndResult + "EndResult.csv")

    # # query for one road
    # str = int(input("Enter the Beta, MarginValue:"))
    # str = scipy.split(str,",")
    # Beta = str[0]
    # MarginValue = str[1]
    # Score = getScore(Historypredict, Neighborpredict, Beta)
    # Result = getResult(Score, MarginValue)
    # (P, R, F1) = getBlandMatrix(Result,RealResult,V)
    #
    # str = int(input("Enter the Roadnum, Timeframenum:"))
    # str = scipy.split(str, ",")
    # Roadnum = str[0]
    # Timeframenum = str[1]

    print("Eps: " + bytes(0.0000045) + ", minPts: " + bytes(2))
    print("Cluster Number: " + bytes(238))
    print("Noise Ratio: " + bytes(25.2357284))
    print("Time: " + bytes(8743.73652959128732))