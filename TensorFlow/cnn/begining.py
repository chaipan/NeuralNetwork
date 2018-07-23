import tensorflow as tf

filter = tf.get_variable("weights", shape=[5,5,3,16], initializer=tf.truncated_normal_initializer)
biases = tf.get_variable("biases", shape=[16], initializer=tf.constant_initializer(0.1))

# 一层卷积计算
conv = tf.nn.conv2d(input="", filter=filter, strides=[1], padding="SAME")
# 加上偏置
bias = tf.nn.bias_add(conv, bias=biases)
# 做一层relu
actived_conv = tf.nn.relu(bias)

"""
    池化层：有效缩减矩阵尺寸，可以加快计算也可以防止过拟合问题
    池化分为最大化池化， 平均池化层， 
"""
# 做一层池化层
"""
 value：[batch, height, width, channels]
 ksize: 每一个维度上池化窗口的数字[1, height, width, 1],维度对应的是输入张量的数字
 strides：窗口在每一个维度上滑动的步长（可以实现维度融合，比如将channel设为3可以压缩channel，
 但一般pool前面的输入是feature map，经过卷积后channel本身为1），一般取[1, stride,stride, 1]
 返回：[batch, height, width, channels]
 的tensor
"""
pool = tf.nn.max_pool(actived_conv, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")








