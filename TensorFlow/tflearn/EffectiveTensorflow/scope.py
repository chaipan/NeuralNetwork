import TensorFlow as tf

a = tf.constant(1)
print(a.name)


b = tf.Variable(1, name="b")
b_2 = tf.Variable(1, name="b")
print(b.name)
print(b_2.name)


a = tf.constant(1, name="a")
a_2 = tf.constant(1, name="a")
print(a.name)
print(a_2.name)

c = tf.get_variable(name="c", shape=[1])
# c_2 = tf.get_variable(name="c", shape=[1]) 这种方法不被允许

"""
    TensorFlow.name_scope影响tf.Variable的域名称，但不会影响tf.get_variable的
    TensorFlow.variable_scope会影响tf.get_variable的名称
"""


with tf.name_scope("scope1"):
    c = tf.constant(1, name="c")
    print(c.name)
