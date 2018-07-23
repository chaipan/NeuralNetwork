import TensorFlow as tf
import numpy as np
from numpy.random import RandomState
"""
    构建一个两个隐藏层，第一层两个节点，第二层三个节点的简单神经网络。
    输入为数据形状为X[n,2]，n为batch大小，w1[2,3], w2[3,1],输出层
"""


batch_size = 8
steps = 1000


w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1), name='w1')
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1), name='w2')

x = tf.placeholder(shape=[None, 2], name='x', dtype=tf.float32)
y_ = tf.placeholder(shape=[None, 1], name='y', dtype=tf.float32)

a = tf.matmul(x, w1, name='matmul_a')
y = tf.matmul(a, w2, name='matmul_y')

# 定义损失函数
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1)))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cross_entropy)

# 生成模拟训练数据
rdm = RandomState(1)
data_size = 128
X = rdm.rand(data_size, 2) # 数据生成略显粗糙
np.random.rand()
Y = [[int(x1 + y1 < 1 )] for (x1, y1) in X]
Y = np.asarray(Y)
print("x.shape{0},y.shape{1}".format(X.shape, Y.shape))
# 创建会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("w1 init value:{0}".format(sess.run(w1)))
    print("w2 init value:{0}".format(sess.run(w2)))
    # 设定训练次数
    for i in range(steps):
        # 选定batch_seze大小的训练样本
        start = (i * batch_size) % data_size
        # 可能出现中间某些批次小于batch_seze的情况，但是下一次又不是头数据集开头计算的情况
        end = min(start + batch_size, data_size)
        sess.run(optimizer, feed_dict={x:X[start:end, :], y_:Y[start:end]})
        print(sess.run(cross_entropy, feed_dict={x:X, y_:Y}))

    print("trained w1:{0}".format(sess.run(w1)))
    print("trained w2:{0}".format(sess.run(w2)))



