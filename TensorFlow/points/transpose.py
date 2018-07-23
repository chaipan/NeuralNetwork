import tensorflow as tf
import numpy as np


"""
	正常情况下perm的值为【0,1,2】
	代表的是形状数据shape的位置
	当变成【0,2,1】时表示原矩阵要将shape第三个位置上的数据和第二个交换
	比如shape = [2,3,4],当perm=[2,1,0]时执行transpose后变成shape变成[4,3,2]
	
"""

a = np.random.randint(0,10,[2,3,4] )
print(a)
sess = tf.Session()
b = sess.run(tf.transpose(a, perm=[0,2,1]))
print(b)