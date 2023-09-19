import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
cmapTest = ListedColormap(['#00FFFF', '#FF00FF', '#FFFF00'])
iris = datasets.load_iris()
x, y = iris.data, iris.target

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 1234)

print(xTrain.shape)
print(xTrain[0])

print(yTrain.shape)
print(yTrain)

from knn import knn
classify = knn(k = 3)
classify.fit(xTrain, yTrain)
predictions = classify.predict(xTest)
print(xTest)
print(yTest)
accuracy = np.sum(predictions == yTest)/len(yTest)
print(accuracy)

#vcl sao ma no bo di 2 so dau trong data z, no ve co 2 so sau thoi

plt.figure()
plt.scatter(x[:, 2], x[:, 3], c = y, cmap = cmap, edgecolor = 'k', s = 20)
plt.scatter(xTest[:, 2], xTest[:, 3], c = predictions, cmap = cmapTest, edgecolor = 'k', s = 20)
plt.show()

