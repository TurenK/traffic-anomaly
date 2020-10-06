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


def multipeMatrix(filenameW,filenameH,filenameOut):
    W = readCSV(filenameW)
    H = readCSV(filenameH)
    labels = W[:,0]
    Wnew = W[:,1:]
    V = np.column_stack((labels,np.dot(Wnew,H)))
    writeCSV(V,filenameOut)


# def createNormalDistribution(V):
#     for rownum in V.shape[0]:
#



if __name__ == '__main__':
    V = multipeMatrix("C:\\Users\\11877\\Desktop\\temp\\20111122WKM.csv","C:\\Users\\11877\\Desktop\\temp\\20111122H.csv","C:\\Users\\11877\\Desktop\\temp\\20111122V.csv")
    V = readCSV("C:\\Users\\11877\\Desktop\\temp\\20111120V.csv")




