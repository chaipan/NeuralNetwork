import tensorflow as tf

NUM_LABELS = 10
IMAGE_SIEZ = 28
IMAGE_CHANNELS = 1

CONV1_SIZE = 5
CONV1_DEEP = 32

CONV2_SIZE = 5
CONV2_DEEP = 64

FC1_NODE = 512
FC2_NODE = NUM_LABELS

def inference(input_tensor,regularizer):
	with tf.variable_scope("layers1_conv1"):
		conv1_filter = tf.get_variable(name="conv1_filter",shape=[CONV1_SIZE, CONV1_SIZE, IMAGE_CHANNELS, CONV1_DEEP], dtype=tf.float32, initializer=tf.truncated_normal_initializer)
		conv1_biases = tf.get_variable(name="conv1_biases",shape=[CONV1_DEEP], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
		conv1 = tf.nn.conv2d(input_tensor, conv1_filter, [1,1,1,1], padding="SAME") + conv1_biases
		relu1 = tf.nn.relu(conv1)

	with tf.variable_scope("layers2_pool1"):
		conv1_pool1 = tf.nn.max_pool(relu1, [1,2,2,1], [1,2,2,1], padding="SAME")

	with tf.variable_scope("layers3_conv2"):
		conv2_filter = tf.get_variable(name="conv2_filter",shape=[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], dtype=tf.float32, initializer=tf.truncated_normal_initializer)
		conv2_biases = tf.get_variable(name="conv2_biases",shape=[CONV2_DEEP], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
		conv2 = tf.nn.conv2d(conv1_pool1, conv2_filter, [1,1,1,1], padding="SAME") + conv2_biases
		relu2 = tf.nn.relu(conv2)

	with tf.variable_scope("layers4_pool2"):
		conv2_pool2 = tf.nn.max_pool(relu2, [1,2,2,1], [1,2,2,1], padding="SAME")


	with tf.variable_scope("layers5_fc1"):
		# pool2传过来的数据是[BATCH_SIZE, x, x, CONV2_DEEP],在计算全连接层时需要将数据转化成[BATCH_SIZE, cols]的结构
		pool2_shape = conv2_pool2.get_shape().as_list()
		pool2_batch = pool2_shape[0]
		pool2_cols = pool2_shape[1]*pool2_shape[2]*pool2_shape[3]
		pool2_shaped = tf.reshape(conv2_pool2, [pool2_batch, pool2_cols])
		fc1_weight = tf.get_variable(name="fc1_weight",shape=[pool2_cols, FC1_NODE], dtype=tf.float32, initializer=tf.truncated_normal_initializer)
		fc1_biases = tf.get_variable(name="fc1_biases",shape=[FC1_NODE], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
		if regularizer is not None:
			tf.add_to_collection("losses", regularizer(fc1_weight))
		fc1 = tf.matmul(pool2_shaped, fc1_weight) + fc1_biases

	with tf.variable_scope("layer6_fc2"):
		fc2_weight = tf.get_variable(name="fc2_weight",shape=[FC1_NODE, FC2_NODE], dtype=tf.float32, initializer=tf.truncated_normal_initializer)
		fc2_biases = tf.get_variable(name="fc2_biases",shape=[FC2_NODE], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
		fc2 = tf.matmul(fc1, fc2_weight) + fc2_biases
		if regularizer is not None:
			tf.add_to_collection("losses", regularizer(fc2_weight))


	logits = fc2
	return fc2



