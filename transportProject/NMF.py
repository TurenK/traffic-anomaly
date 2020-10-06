import numpy


def readCSV():
    my_matrix = numpy.loadtxt(open("C:\\Users\\11877\\Desktop\\20111120.csv", "rb"), delimiter=",", skiprows=0)
    my_matrix = my_matrix[:, 3:99]
    return my_matrix


def NMFGradientDescent(R, P, Q, K, steps=5000, alpha=0.02, beta=0.2):
    WNew =P
    HNew =  Q
    VTemp = R + 0.001
    for step in range(steps):
        WNew[WNew < 0.001] = 0.001
        HNew[HNew < 0.001] = 0.001
        for i in range(len(VTemp)):
            for j in range(len(VTemp[i])):
                if VTemp[i][j] > 0:
                    eij = VTemp[i][j] - numpy.dot(WNew[i, :], HNew[:, j])
                    for k in range(K):
                        WNew[i][k] = numpy.around(WNew[i][k] + alpha * (2 * eij * HNew[k][j] - beta * WNew[i][k]),decimals=4)
                        HNew[k][j] = numpy.around(HNew[k][j] + alpha * (2 * eij * WNew[i][k] - beta * HNew[k][j]),decimals=4)
        e = 0
        for i in range(len(VTemp)):
            for j in range(len(VTemp[i])):
                if VTemp[i][j] > 0:
                    e = e + pow(VTemp[i][j] - numpy.dot(WNew[i, :], HNew[:, j]), 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(WNew[i][k], 2) + pow(HNew[k][j], 2))
        if e < 0.001:
            print("iterate end")
            break
    return WNew, HNew


def NMFNormal(V, W, H, steps=5000):
    WNew = W
    HNew = H
    VTemp = V + 0.00000001
    for step in range(steps):
        WNew[WNew < 0.00000001] = 0.00000001
        HNew[HNew < 0.00000001] = 0.00000001
        WNew = numpy.multiply(WNew, numpy.true_divide(numpy.dot(VTemp, HNew.T), numpy.dot(WNew, numpy.dot(HNew, HNew.T))))
        HNew =  numpy.multiply(HNew, numpy.true_divide(numpy.dot(WNew.T, VTemp), numpy.dot(numpy.dot(WNew.T, WNew), HNew)))
        VNew = numpy.dot(WNew, HNew)
        e = numpy.linalg.norm(V - VNew)
        if e < 10:
            print("NMF End")
            break
    return WNew, HNew

if __name__ == '__main__':
    # print(readCSV())
    # V = numpy.random.randint(0,1, size=[4, 4])
    V = readCSV() * 1.0
    print("Finish Reading")
    W = numpy.random.randint(1, 100, size=[V.shape[0], 4]) * 1.0
    H = numpy.random.randint(1, 100, size=[4, V.shape[1]]) * 1.0
    (W, H) = NMFGradientDescent(V, W, H, 4)
    #(W, H) = NMFNormal(V, W, H)
    numpy.savetxt('C:\\Users\\11877\\Desktop\\20111120W.csv', W, delimiter=',')
    numpy.savetxt('C:\\Users\\11877\\Desktop\\20111120H.csv', H, delimiter=',')
