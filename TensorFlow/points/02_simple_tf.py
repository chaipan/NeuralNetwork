import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import TensorFlow as tf

# example1:简单方式创建log文件写出
a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')
writer = tf.summary.FileWriter('./graphs/simple',tf.get_default_graph())
with tf.Session() as sess:
    print(sess.run(x))
writer.close()

# example2: div魔法
a = tf.constant([2,2], name='a')
b = tf.constant([[0,1],[2,3]],name='b')
with tf.Session() as sess:
    print(sess.run(tf.div(b, a)))
    print(sess.run(tf.divide(b,a)))
    print(sess.run(tf.truediv(b, a)))
    print(sess.run(tf.floordiv(b, a)))
    print(sess.run(tf.truncatediv(b, a)))
    print(sess.run(tf.floor_div(b, a)))

# examples3: 张量乘法
a = tf.constant([10,20],name='a')
b = tf.constant([2,3], name='b')
with tf.Session() as sess:
    print(sess.run(tf.multiply(a,b)))
    print(sess.run(tf.tensordot(a,b,axes=1)))

# example4: python 本地类型
t_0 = 9
x = tf.zeros_like(t_0)
y = tf.zeros_like(t_0)

t_1 = ['apple','peach','banana']
x = tf.zeros_like(t_1)
print(tf.get_default_graph().as_graph_def())


