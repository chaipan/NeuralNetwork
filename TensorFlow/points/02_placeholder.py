import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import TensorFlow as tf

a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant(value=[5,5,5],dtype=tf.float32)
c = tf.add(a, b)

#开启tensorBoard
writer = tf.summary.FileWriter('./graphs',tf.get_default_graph())
with tf.Session() as sess:
    print(b)
    print(sess.run(c, {a:[1,1,1]}))
writer.close()


