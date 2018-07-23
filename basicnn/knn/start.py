from basicnn.knn import knn
from utils import load
import numpy as np


Xtr,Ytr,Xte,Yte = load.load()
knn =  knn()
knn.train(Xtr,Ytr,None,None,Xte,Yte)
labels = knn.predict(Xte[0:100,],5)
labelss = knn.predicts(Xte[0:100,])
print("{}%".format(np.mean(np.array(labels) == Yte[0:100])*100))
print("{}%".format(np.mean(labels == labelss)*100))