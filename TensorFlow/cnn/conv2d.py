import tensorflow as tf


"""
    函数的作用是做一层卷积， 
    def conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None):
    input:4-d [batch, in_height, in_width, in_channels]
    fileter: 4-d`[filter_height, filter_width, in_channels, out_channels]`
    stride:1-d
    padding:A `string` from: `"SAME", "VALID"`.
"""


conv = tf.nn.conv2d()