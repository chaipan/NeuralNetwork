import tensorflow as tf

INPUT_NODE = 28*28
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5


FC_SIZE = 512

def inference(input_tensor, train, regularizer):
    # 卷积层1， 输入为28*28*1， 卷积核为5*5*32，步调为1，不填充，输出为24*24*32。池化为2*2， 输出为12*12*32
    # truncated_normal_initializer截断的正太分布， 当生成的数据大于期望两个标准差则丢弃，重新生成数据
    with tf.variable_scope("layer1_conv1"):
        conv1_weights = tf.get_variable(name="weights",
                                        shape=[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
                                        dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer)
        conv1_biases = tf.get_variable(name="biases",
                                       shape=[CONV1_DEEP],
                                       dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.1))
        # stride步调，在输入各维度上的步调[batch, height, width, channels],一般取[1,stride,stride,1s]
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, [1,1,1,1], padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.variable_scope("layer2_pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME" )

    with tf.variable_scope("layer3_conv2"):
        conv2_weights = tf.get_variable(name="weights",
                                        shape=[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                        dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer)
        conv2_biases = tf.get_variable(name="biases", shape=[CONV2_DEEP], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, filter=conv2_weights, strides=[1,1,1,1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.variable_scope("layer4_pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
        # 在使用全连接层前，将输入压缩成一维数据
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope("layer5_fc1"):
        fc1_weights = tf.get_variable(name="weights", shape=[nodes, FC_SIZE], dtype=tf.float32, initializer=tf.truncated_normal_initializer)
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(fc1_weights))
        fc1_biases = tf.get_variable(name="biases", shape=[FC_SIZE], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train is not None:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit









