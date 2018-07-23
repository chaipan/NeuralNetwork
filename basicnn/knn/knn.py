import numpy as np
import operator as op
class knn(object):
    def __init__(self):
        pass



    def train(self,X, Y, Xva, Yva, Xte, Yte):
        self.Xtr = X
        self.Ytr = Y
        self.Xva = Xva
        self.Yva = Yva
        self.Yte = Yte

    def predict(self,Xte,k=5):
        # cifar10的数据集中图片数据为ndarray，标签为list
        num_test = Xte.shape[0]
        labels = []
        for i in range(num_test):
            # 使用L2的欧式距离
            distances = np.sqrt(np.sum(np.square(self.Xtr - Xte[i]), axis=1))
            labels.append(self.statLabel(distances, k))
        return labels

    def statLabel(self,distances,k):
        #初始化一个字典
        labels = {}
        for i in range(10):
            labels.setdefault(i,0)
        #k个最小的距离
        distances = list(distances)
        min_indexs = []
        for i in range(k):
            min_index = np.argmin(distances)
            distances[min_index] = 0
            min_indexs.append(min_index)
        #对k个值进行统计，求出出现最多的标签
        for min_index in min_indexs:
            labels[self.Ytr[min_index]] += 1
        label = sorted(labels.items(), key=lambda d: d[1],reverse=True)
        return label[0][0]

    def predicts(self,X):
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



