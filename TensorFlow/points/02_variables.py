import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
"""
- 0：显示所有日志（默认等级） 
- 1：显示info、warning和error日志 
- 2：显示warning和error信息 
- 3：显示error日志信息 
"""
import TensorFlow as tf
import numpy as np

# example1: creating variables

# Example 1: creating variables

s = tf.Variable(2, name='scalar')
m = tf.Variable([[0, 1], [2, 3]], name='matrix')
W = tf.Variable(tf.zeros([784,10]), name='big_matrix')
V = tf.Variable(tf.truncated_normal([784, 10]), name='normal_matrix')

s = tf.get_variable('scalar', initializer=tf.constant(2))
m = tf.get_variable('matrix', initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable('big_matrix', shape=(784, 10), initializer=tf.zeros_initializer())
V = tf.get_variable('normal_matrix', shape=(784, 10), initializer=tf.truncated_normal_initializer())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(V.eval())

# example2:assigning values to variables
w = tf.Variable(10)
w.assign(100)
with tf.Session() as sess:
    sess.run(w.initializer)
    print(sess.run(w))
# 此处的w不会显示指定的值，因为在会话中没有执行指定操作。
with tf.Session() as sess:
    sess.run(w.assign(100))
    print(sess.run(w))

w = tf.Variable(10)
with tf.Session() as sess:
    sess.run(w.initializer)
    print(sess.run(w.assign_add(10)))
    print(sess.run(w.assign_sub(2)))

# examples3 每个session都有自己保存的变量
a = tf.Variable(10)
b = tf.Variable(20)
sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(a.initializer)
sess2.run(b.initializer)
print(sess1.run(a.assign_add(100)))
print(sess2.run(b.assign_sub(100)))













