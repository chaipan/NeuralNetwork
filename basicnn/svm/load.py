"""
装载cifar10数据集
"""
import pickle
import gzip

def load():
    ftr = open('D:\cs231n\cifar-10-python\cifar-10-batches-py\data_batch_1', 'rb')
    dictr = pickle.load(ftr, encoding='bytes')
    Xtr = dictr[b'data']
    Ytr = dictr[b'labels']

    return (Xtr, Ytr)