import TensorFlow as tf


a = tf.placeholder(tf.float32, [None, 128])

print(a.shape.as_list())

# return a tensor which represents the shape of given tensor
print(tf.shape(a))

# The static shape of a tensor can be set with Tensor.set_shape() method:
a.set_shape([1,128])

print(a.shape)
a.set_shape([None, 128])

# You can reshape a given tensor dynamically using tf.reshape function:
# reshape不能随便改变形状，元素总数不能改变，必须能被行或者列总除
a = tf.reshape(a, [2,64])
print(a.shape)

# 写一个get_shape函数， 获取张量的静态形状，如果没有静态形状则返回静态形状，没有则返回动态
def get_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    zip_a = zip(static_shape, dynamic_shape)
    for i in zip_a:
        print(i)
    dims = [s[1] if s[0] is None else s[0]
        for s in zip(static_shape, dynamic_shape)
    ]
    return dims

b = tf.placeholder(name="b", shape=[None, 10, 3], dtype=tf.float32)
shape = get_shape(b)
b = tf.reshape(b, shape=[shape[0], shape[1] * shape[2]])
print(b)



if __name__ == "__main__":
    print("-------------------------------------------------")

