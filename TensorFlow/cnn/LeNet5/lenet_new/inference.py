import tensorflow as tf
import tensorflow.contrib as contrib

CONV1_SIZE = 5
CONV1_DEEP = 32
CONV1_STRIDE = 1

CONV2_SIZE = 5
CONV2_DEEP = 32
CONV2_STRIDE = 1

INPUT_CHANNEL = 1

FC1_NODES = 512
FC2_NODE = CLASS_NUMS = 10




def inference(input):
	# 卷积层的输入是[batch, height, width, channel]
	with tf.variable_scope("conv1"):
		conv1_weight = tf.get_variable(name="conv1_weight",
									   shape=[CONV1_SIZE, CONV1_SIZE, INPUT_CHANNEL, CONV1_DEEP],
									   dtype=tf.float32,
									   initializer=tf.truncated_normal_initializer)
		conv1_biases = tf.get_variable(name="conv1_biases",
									   shape=[CONV1_DEEP],
									   dtype=tf.float32,
									   initializer=tf.constant_initializer(0.1))
		conv1 = tf.nn.conv2d(input=input,
							 filter=conv1_weight,
							 strides=[1,1,1,1],
							 padding="SAME")
		conv1_relu = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
		conv1_pool = tf.nn.max_pool(value=conv1_relu,
									ksize=[1,2,2,1],
									strides=[1,2,2,1],
									padding="SAME")

	with tf.variable_scope("conv2"):
		conv2_weight = tf.get_variable(name="conv2_weight",
									   shape=[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
									   dtype=tf.float32,
									   initializer=tf.truncated_normal_initializer)
		conv2_biases = tf.get_variable(name="conv2_biases",
									   shape=[CONV2_DEEP],
									   dtype=tf.float32,
									   initializer=tf.constant_initializer(0.1))
		conv2 = tf.nn.conv2d(input=conv1_pool,
							 filter=conv2_weight,
							 strides=[1,1,1,1],
							 padding="SAME")
		conv2_relu = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
		conv2_pool = tf.nn.max_pool(conv2_relu,
									ksize=[1,2,2,1],
									strides=[1,2,2,1],
									padding="SAME")

	with tf.variable_scope("fc1"):
		#此时网络的输出为[batch, 28,28,conv2_filter_count]大小的数据，在接入全连接层时必须转变形状。
		conv2_pool_shape = conv2_pool.get_shape().as_list()
		batch = input.get_shape().as_list()[0]
		column = conv2_pool_shape[1]*conv2_pool_shape[2]*conv2_pool_shape[3]

		fc1_shaped_input = tf.reshape(tensor=conv2_pool, shape=[batch, column])
		fc1_weight = tf.get_variable(name="fc1_weight",
									 shape=[column,FC1_NODES],
									 dtype=tf.float32,
									 initializer=tf.truncated_normal_initializer)
		fc1_biases = tf.get_variable(name="fc1_biases",
									 shape=[FC1_NODES],
									 dtype=tf.float32,
									 initializer=tf.constant_initializer(0.1))

		fc1 = tf.matmul(fc1_shaped_input, fc1_weight)
		fc1_relu = tf.nn.relu(tf.nn.bias_add(fc1, fc1_biases))

		# fc1全连接层节点多，参数数量大，可能出现过拟合，
		tf.add_to_collection("losses", contrib.layers.l2_regularizer(fc1_weight))

	with tf.variable_scope("fc2"):
		fc2_weight = tf.get_variable(name="fc2_weight",
									 shape=[FC1_NODES,FC2_NODE],
									 dtype=tf.float32,
									 initializer=tf.truncated_normal_initializer)
		fc2_biases = tf.get_variable(name="fc2_biases",
									 shape=[FC2_NODE],
									 dtype=tf.float32,
									 initializer=tf.constant_initializer(0.1))

		fc2 = tf.matmul(fc1_relu, fc2_weight)
		fc2 = tf.nn.bias_add(fc2, fc2_biases)

		return fc2


