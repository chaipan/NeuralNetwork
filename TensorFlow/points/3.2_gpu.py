import TensorFlow as tf

g = tf.Graph()

with g.device('/gpu:0'):
    with tf.Session() as sess:
        a = tf.get_variable('a', shape=[1], initializer=tf.zeros_initializer, dtype=tf.int32)
        tf.global_variables_initializer().run()
        print(sess.run(a))
    print("sss")
