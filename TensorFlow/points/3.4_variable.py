import TensorFlow as tf

weights = tf.Variable(initial_value=tf.random_normal([2,3]), name='weights')
biases = tf.Variable(tf.zeros([2,1]))
print(weights.initial_value)
print(weights.initialized_value())

# 也可以使用其他变量的初始化值来初始化新的变量

w1 = tf.Variable(weights.initialized_value(), name='w1')
w2 = tf.Variable(weights.initial_value * 2, name='w2')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(weights))
    print(sess.run(w1))
    print(w2.eval())
