import numpy as np

def signf(w, x):
    if np.sign(np.dot(w.T, x)) >= 0:
        return 1
    return -1

def isConverged(x, y, w):
    for i in range(x.shape[0]):
        xi = x[i].reshape(x.shape[1],1)
        if signf(w, xi) != y[i]:
            return 0
    return 1

class perceptron:
    def fit(X, y):
        #so vector n + so chieu d
        n = X.shape[0]
        xtemp = np.c_[np.ones(n), X]
        d = xtemp.shape[1]
        #khoi tao vector w
        w = [np.random.randn(d,1)]
        ytemp = y.copy()
        for i in range(n):
            if ytemp[i] == 0:
                ytemp[i] = -1
            ytemp[i] *= -1
        misclassifiedPoints = []
        while 1:
            #xep du lieu ngau nhien
            randomId = np.random.permutation(n)
            #duyet qua du lieu
            for i in range(n):
                xi = xtemp[randomId[i], :].reshape(d,1)     #dua vector xi ve vector cot
                yi = ytemp[randomId[i]]     #nhan
                if signf(w[-1], xi) != yi:               #tich trong khac nhan
                    misclassifiedPoints.append(randomId[i])
                    w_star = w[-1]+yi*xi                    #cap nhat
                    w.append(w_star)
            if isConverged(xtemp, ytemp, w[-1]):
                break
        return w

