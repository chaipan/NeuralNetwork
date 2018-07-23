import TensorFlow as tf


steps = 1

learning_rating = tf.train.exponential_decay(
    learning_rate=0.001, global_step=steps, decay_rate=0.1, decay_steps=100, staircase=True, name="learning_rate")


