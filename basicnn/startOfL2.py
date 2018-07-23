from utils import load
from basicnn import NearestNeighborOfL2
import numpy as np


Xtr,Ytr,Xte,Yte = load.load()
nn = NearestNeighborOfL2.NearestNeighbor()
nn.train(Xtr,Ytr)
Ypred = nn.predict(Xte[0:100])

print("accrrancy={}%".format(np.mean(Yte[0:100] == Ypred)*100))
