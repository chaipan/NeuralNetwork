import numpy as np
import matplotlib.pylab as plt

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
# 创建第一个画板
plt.figure(1)
# 第一个画板的第一个子图
plt.subplot(211)
plt.title("211")
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
# 第二个子图
plt.subplot(212)
plt.title("212")
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()

