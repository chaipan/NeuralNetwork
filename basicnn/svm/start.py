from basicnn.svm import load
from basicnn.svm import lossi
import numpy as np

Xtr, Ytr = load.load()
x = Xtr[0:1,].reshape(Xtr.shape[1],1)
y = Ytr[0]

w = np.zeros((10, x.shape[0]),float)

print(lossi.L_i(x, y, w, 1))
print(lossi.L_i_vector(x, y, w, 1))




