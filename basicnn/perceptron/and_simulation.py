import numpy as np

"""
使用单层感知机模拟逻辑与
0 0 0
1 0 0
0 1 0
1 1 1
"""
W = np.random.rand(3)
def single_perceptron(input):
    x = np.reshape(input, [2, 1])
    np.append(x, [[-1]], 0)
    z = np.dot(W, x)
    return sgn(z)


def sgn(x):
    y = 1 if x >= 0 else -1
    return y

def train():
    pass

