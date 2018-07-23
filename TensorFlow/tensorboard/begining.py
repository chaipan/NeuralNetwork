import TensorFlow as tf


a = tf.Variable(initial_value=1.1)
b = tf.Variable(initial_value=1.2)

c = tf.add(a, b)
"""
    创建writer
"""
writer = tf.summary.FileWriter(logdir="graph/begining/", graph=tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(c)

writer.close()

"""
    启动命令tensorboard --logdir="D://GItHub//cs231n-master//Image Classification//NeuralNetwork//TensorFlow//tensorboard//model//begining"
    访问：http://localhost:6006 默认6006端口，可在启动命令中加入port设置
"""