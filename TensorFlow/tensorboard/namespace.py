import TensorFlow as tf

with tf.variable_scope("foo"):
    a = tf.get_variable("a",shape=[1])
    print(a.name)

with tf.variable_scope("bar"):
    b = tf.get_variable("b", shape=[2])
    print(b.name)

"""
    namescope中使用get_variable不会添加空间名称
    使用Variable会添加空间名称
"""
with tf.name_scope("a"):
    c = tf.get_variable("c", shape=[1])
    print(c.name)
    e = tf.Variable(initial_value=0.1)
    print(e.name)

with tf.name_scope("b"):
    d = tf.get_variable("c", shape=[1])
    print(d.name)
    """
        ValueError: Variable c already exists, disallowed. Did you mean to set reuse=True in VarScope? Originally defined at:
    """