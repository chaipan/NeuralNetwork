import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt


def a():
    mon = [range(120)]
    a = np.power((1+0.0079741404289038), mon)
    # np.extract()/
    plt.scatter(mon, a-1)
    plt.show()

if __name__ == "__main__":
    print(0.09568968514684562/365)
    # a()