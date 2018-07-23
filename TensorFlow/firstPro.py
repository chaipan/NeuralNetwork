import TensorFlow as tf

a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a,b)
with tf.Session() as sess:
    print(sess.run(c))
    writer = tf.summary.FileWriter('./graphs', sess.graph)
writer.close()