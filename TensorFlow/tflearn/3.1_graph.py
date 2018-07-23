import TensorFlow as tf

# 验证默认图
a = tf.constant(value=2, name='a')
print(a.graph is tf.get_default_graph())

# 创建新的计算图，每个计算图之间是隔离的
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable('v', initializer=tf.zeros_initializer(), shape=[1])

g2 = tf.Graph()
with g2.as_default():
    w = tf.get_variable('v', initializer=tf.ones_initializer(), shape=[1])

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    # with tf.variable_scope("", reuse=True):
    print(sess.run(v))

with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    print(sess.run(w))