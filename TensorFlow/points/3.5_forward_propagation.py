import TensorFlow as tf

"""
    向前传播算法：单层感知机（学名似乎没有向前传播算法）
"""
w1 = tf.Variable(initial_value=tf.random_normal([2,3], stddev=1, seed=1), name='w1', trainable=True)
w2 = tf.Variable(initial_value=tf.random_normal([3,1], stddev=1, seed=1), name='w2', trainable=False)

x = tf.constant(value=[[0.7, 0.9]])

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

writer = tf.summary.FileWriter(logdir='D:/logs/graph/3.5', graph=tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(y))
    print("------all variables------")
    # 所有的变量存放于GraphKeys.VARIABLE当中， 使用tf.all_variables()方法可以获取到，是一个tensor列表
    print(tf.all_variables())

    """ 可以通过设置变量中的trainable参数将变量设置为可训练的参数。可训练的参数存放于tf.GraphKeys.TRAINABLE_VARIABLES当中，
    可以通过tf.trainable_variables()获取"""
    print(tf.trainable_variables())
"""
    从tensorboard中可以看出Variable过程是一个运算的过程，是初始化后的运算节点。
"""
