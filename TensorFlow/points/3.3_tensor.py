import TensorFlow as tf

"""
    tensor是tf中放入张量，图是tf的计算载体。
    tf中的张量并没有保存运算结果，他保存的是计算过程。
"""

a = tf.constant(0, dtype=tf.int32, name='a', shape=[1])
b = tf.constant(1, dtype=tf.int32, name='b', shape=[1])
add_1 = tf.add(a, b, name='add_1')
add_2 = tf.add(a, b, name='add_1')
print(add_1)
print(add_2)

"""
    Tensor("add_1:0", shape=(), dtype=int32)
    Tensor("add_1_1:0", shape=(), dtype=int32)
    由上面可以看出，一个张量主要保存三个属性：名字，维度和类型
    名字：张量的唯一标识符。写出了张量是如何计算得来的。计算图中的每一个节点代表了一个计算，而计算结果就保存在张量。
    名字的格式为：node:src_output:node为节点，src_output为节点的第几个输出。
"""
# 获取张量的值：
"""
   上面给出了张量保存的只是计算过程，所以要得出张量的值就必须要计算图中的节点，要计算图，需要引入回话Session
"""
# 第一种方式可以使用eval函数
with tf.Session() as sess:
    print(add_1.eval())

with tf.Session() as sess:
    print(sess.run(add_1))

# 或者下面这种形式也可以
sess = tf.Session()
# print(add_1.eval())
"""
    Cannot evaluate tensor using `eval()`: No default session is registered. 
    Use `with sess.as_default()` or pass an explicit session to `eval(session=sess)`
    要计算张量必须要使用回话
"""
print(add_1.eval(session = sess))