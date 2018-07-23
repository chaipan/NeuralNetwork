import numpy as np


class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, Y):
        self.Xtr = X
        self.Ytr = Y

    def predict(self,X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test)
        for i in range(0, num_test):
            sum = np.sum(np.square(self.Xtr - X[i]),1)
            print("sum={}".format(sum))
            distances = np.sqrt(sum)
            min_index = np.argmin(distances)
            Ypred[i] = self.Ytr[min_index]
            print("{}th finished".format(i))
        return Ypred







