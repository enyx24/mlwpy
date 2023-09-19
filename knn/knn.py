import numpy as np
from collections import Counter

def distance(x, y):
    return np.sqrt(np.sum((x-y)**2))

class knn:
    def __init__(self, k):
        self.k = k
    def fit(self, x, y):
        self.xTrain = x
        self.yTrain = y
    def predict(self, xTest):
        return [self._predict(x) for x in xTest]
        
    def _predict(self, x):
        distances = [distance(x, xtrain) for xtrain in self.xTrain]
        kNeighborsIndices = np.argsort(distances)[:self.k]
        kNeighborsLabels = [self.yTrain[i] for i in kNeighborsIndices]
        return Counter(kNeighborsLabels).most_common(1)[0][0]


print('kwegud')

