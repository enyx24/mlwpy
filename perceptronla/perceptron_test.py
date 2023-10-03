#import thu vien numpy
import numpy as np
#import datasets tu thu vien scikit_learn
from sklearn import datasets
#import ham chia du lieu train-test
from sklearn.model_selection import train_test_split
#import thu vien matplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#map mau cho data
cmap = ListedColormap(['#0000FF', '#FF0000'])

#load iris dataset
iris = datasets.load_iris()
x, y = iris.data, iris.target

#chi lay 2 lop de dam bao linearly separable
x = x[:100]
y = y[:100]

#chi lay 2 feature sepal length va sepal width
x = x[:, 0:2]

#import perceptron
from perceptron import perceptron
w = perceptron.fit(x, y)

#ve duong thang
def drawLines(w):
    w0, w1, w2 = w[0], w[1], w[2]
    if w2 != 0:
        x11, x12 = -100, 100
        return plt.plot([x11, x12], [-(w1*x11+w0)/w2, -(w1*x12+w0)/w2], 'k')
    else:
        x10 = -w0/w1
        return plt.plot([-100, 100], [x10, x10], 'k')

plt.figure()
plt.axis([4.1, 7.1, 1.9, 4.5])
plt.scatter(x[:, 0], x[:, 1], c = y, cmap = cmap, edgecolor = 'k', s = 20)
print("So lan cap nhat: ",len(w))
drawLines(w[-1])  
plt.show()

