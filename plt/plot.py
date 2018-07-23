import matplotlib.pylab as plt
import numpy as np

# 如果给plot提供单个列表或者数组，plot默认是y值序列，且自动生成x值序列， x值按长度为y序列的长度，步调为1生成
def plot_single():
    plt.plot([1,2,3,5])
    plt.ylabel("y")
    plt.show()

def plot_double():
    plt.plot([1,2,3,4,5], [1,4,9,16,25])
    plt.ylabel("y")
    plt.show()

def line_color():
    # https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
    plt.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25], 'ro')
    # axis表示[xmin，xmax，ymin，ymax]
    plt.axis([0, 6, 0, 20])
    plt.show()

def multi_lines():
    t = np.arange(0, 10, 0.1)
    # t**2 bs表示蓝色的线条，是平方标志
    plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    plt.show()

def line_property():
    #linewidth

    line,  = plt.plot(np.random.randint(0,10,size=20), linewidth = 4.0)
    line.set_antialiased(False)  # turn off antialising
    plt.show()


if __name__ == '__main__':
    multi_lines()


