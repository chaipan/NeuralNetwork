import TensorFlow as tf

'''
TensorFlow.Variable() ：局部变量。
    特点：永远生成新的变量，不会与已存在的variable重名
TensorFlow.name_scope()：局部空间。
    特点：永远生成新的命名空间，不会与已存在的name_scope、variable_scope冲突 ，也就是说name_scope也是局部的、临时的
TensorFlow.get_variable()：全局变量。
    特点：可共享，不能重复，解决了众多方法调用时 参数来回传递的困境，
 TensorFlow.variable_scope()：全局空间。
    特点：跟get_variable()性质雷同，如果当前已存在同名variable_scope，则重复使用已存在的。
'''


# name_scope
# name_scope的作用是定义类似于命名空间的区域，在不同的区域中可以定义同名的变量，不同区域的变量不能共享
with tf.name_scope("name_scope_1"):
    a = tf.constant('0', name='a')
    b = tf.constant('0', name='b')

with tf.name_scope("name_scope_2"):
    c = tf.constant('1', name='a')
    d = tf.constant('1', name='b')

print(a.name)
print(b.name)
print(c.name)
print(d.name)

# variable_scope
# 在不同的变量区域定义同名变量不会重复,getVariable的规则是有则拿来（同scope），无则创建
with tf.variable_scope('variable_scope_1'):
    variable_a = tf.get_variable('a', shape=[1])
    variable_b = tf.get_variable('b', shape=[1])

with tf.variable_scope('variable_scope_2'):
    variable_c = tf.get_variable('a', shape=[1])
    variable_d = tf.get_variable('b', shape=[1])
print(variable_a.name)
print(variable_b.name)
print(variable_c.name)
print(variable_d.name)

# 如果需要将已经定义过得区域变量在其他地方使用，则需要设置reuse=True，并且scope同名
with tf.variable_scope('variable_scope_2', reuse=True):
    variable_e = tf.get_variable('a', shape=[1])
    print(variable_e.name)

# 如果需要将已经定义过得区域变量在其他地方使用，则需要设置reuse=True，并且scope同名
with tf.variable_scope('variable_scope_2'):
    print("不设置reuse {0}")
    variable_e = tf.get_variable('b', shape=[1])
    print("不设置reuse {0}".formate(variable_e.name))
    '''
    ValueError: Variable variable_scope_2/b already exists, disallowed. 
    Did you mean to set reuse=True in VarScope? Originally defined at:
    同scope且同名变量，不设置resue=True会报出上面的错误信息
    '''

# 如果使用了reuse=True但是变量并没有在之前定义，会报出下面的错误
with tf.variable_scope('variable_scope_2', reuse=True):
    # variable_e = tf.get_variable('c', shape=[1])
    # print(variable_e.name)
    """
    ValueError: Variable variable_scope_2/c does not exist, 
    or was not created with TensorFlow.get_variable(). Did you mean to set reuse=None in VarScope?
    """

# Variable不能创建共同变量
with tf.variable_scope('variable_scope_2', reuse=True):
    variable_f = tf.get_variable('a', shape=[1])
    variable_g = tf.Variable(0, name='a')
    print("---------------------分隔符---------------------")
    print(variable_f.name)
    print(variable_g.name)