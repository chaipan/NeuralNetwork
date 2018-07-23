import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import TensorFlow as tf
#####################################################
# NORMAL LOADING                                    #
#####################################################
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs',sess.graph)
    print(tf.get_default_graph().as_graph_def())
    '''for _ in range(10):
        sess.run(z)
        print(TensorFlow.get_default_graph().as_graph_def())
        writer.close()'''

#####################################################
# LAZY LOADING
# 延迟加载时有问题，就是每次执行sess.run(tf.add(x,y,name='z'))时都会初始化一遍匿名变量，若循环执行多次就会初始化多次，执行速度远远慢于非延迟加载#
#####################################################
x = tf.Variable(10,name='x')
y = tf.Variable(10, name='y')
with tf.Session() as sess:
    sess.run(tf.add(x,y,name='z'))



