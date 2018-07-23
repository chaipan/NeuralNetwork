import TensorFlow as tf

with tf.name_scope("input1"):
    input1 = tf.constant(value=[1.0,1.0,1.0], name="input1")

with tf.name_scope("input2"):
    input2 = tf.Variable(initial_value=tf.random_uniform([3]), name="input2")

out_put = tf.add(input1, input2)

writer = tf.summary.FileWriter("graph/beginning2", graph=tf.get_default_graph())
writer.close()