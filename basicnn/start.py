from basicnn import nn
from utils import load
import numpy as np

Xtr,Ytr,Xte,Yte = load.load()
nn = nn()
nn.train(Xtr,Ytr)
Ypred = nn.predict(Xte[0:100])
print("accuracy is {}%".format(np.mean(Yte[0:100] == np.array(Ypred,dtype=int))*100))
