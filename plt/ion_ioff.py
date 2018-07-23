import matplotlib.pylab as plt
from PIL import Image
import numpy as np
"""
介绍
在使用matplotlib的过程中，发现不能像matlab一样同时开几个窗口进行比较，于是查询得知了交互模式，
但是放在脚本里运行的适合却总是一闪而过，图像并不停留，遂仔细阅读和理解了一下文档，记下解决办法，问题比较简单，仅供菜鸟参考。
python可视化库matplotlib有两种显示模式：
阻塞（block）模式
交互（interactive）模式
"""
# 在Python Consol命令行中，默认是交互模式。而在python脚本中，matplotlib默认是阻塞模式。
"""
其中的区别是:
在交互模式下：
plt.plot(x)或plt.imshow(x)是直接出图像，不需要plt.show()
如果在脚本中使用ion()命令开启了交互模式，没有使用ioff()关闭的话，则图像会一闪而过，并不会常留。要想防止这种情况，需要在plt.show()之前加上ioff()命令。
在阻塞模式下：
打开一个窗口以后必须关掉才能打开下一个新的窗口。这种情况下，默认是不能像Matlab一样同时开很多窗口进行对比的。
plt.plot(x)或plt.imshow(x)是直接出图像，需要plt.show()后才能显示图像
"""

img = Image.open("C://Users//20577//Desktop//cnn_compute.jpg")
img = np.array(img)

# 打开交互模式
plt.ion()
# 同时打开两个窗口显示图片
plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(img, cmap=plt.cm.gray)
# 显示前关掉交互模式，如果不关掉图像会一闪而过
plt.ioff()
# plt.pause(1)可以让图像停顿一秒钟
plt.show()
