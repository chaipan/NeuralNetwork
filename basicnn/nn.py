import numpy as np


class nn(object):
    def __init__(self):
        pass

    def train(self, X, Y):
        self.Xtr = X
        self.Ytr = Y

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = []
        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i]), 1)
            min_index = np.argmin(distances)
            Ypred.append(self.Ytr[min_index])
            print("{}th finished".format(i))
        return Ypred
