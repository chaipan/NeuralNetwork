import TensorFlow as tf



"""
stack是一个打包函数，将不同的矩阵压缩成指定形状的矩阵
"""

x = tf.random_normal(shape=[1,3])
y = tf.random_normal(shape=[1,3])


with tf.Session() as sess:
    z = tf.stack([x, y], axis=0)
    print(sess.run(z))
    """
    [[-0.7234253   0.89249533 -1.2357484 ]
    [ 0.5864474  -1.007866    0.5740551 ]]
    """
    w = tf.stack([x, y], axis=1)
    print(sess.run(w))
    t = tf.stack([x, y], axis=2)
    print(sess.run(t))
