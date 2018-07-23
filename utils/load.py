import pickle
import gzip


def load():
    ftr = open('D:\cs231n\cifar-10-python\cifar-10-batsches-py\data_batch_1','rb')
    dictr = pickle.load(ftr,encoding='bytes')
    Xtr = dictr[b'data']
    Ytr = dictr[b'labels']

    fte = open('D:\cs231n\cifar-10-python\cifar-10-batches-py//test_batch','rb')
    dicte = pickle.load(fte,encoding='bytes')
    Xte = dicte[b'data']
    Yte = dicte[b'labels']

    return (Xtr,Ytr,Xte,Yte)