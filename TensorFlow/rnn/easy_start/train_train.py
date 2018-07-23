import tensorflow as tf
import numpy as np
import TensorFlow.rnn.easy_start.train_inference as inference

BATCH_SIZE = 5
SERIES_NUM = 10

def generate_date():
	x = np.random.randint(0, 10, [1000, SERIES_NUM, inference.NUM_CLASSES])
	y = np.argmax(x, axis=2)
	return x, y

def train(data_x, data_y):
	x_placeholder = tf.placeholder(shape=[BATCH_SIZE, SERIES_NUM, inference.NUM_CLASSES],
								   dtype=tf.float32)
	y_placeholder = tf.placeholder(shape=[BATCH_SIZE, SERIES_NUM],
								   dtype=tf.int32)
	logits = inference.inference(x_placeholder)
	cross_extropy = tf.losses.sparse_softmax_cross_entropy(labels=y_placeholder,
												  logits=logits)
	loss = tf.reduce_mean(cross_extropy)
	optimizer = tf.train.AdamOptimizer(0.01)

	train_op = optimizer.minimize(loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	batch_time = 0
	for i in range( int(1000.0/BATCH_SIZE) ):
		x_feed = data_x[batch_time:batch_time+BATCH_SIZE, :, :]
		y_feed = data_y[batch_time:batch_time+BATCH_SIZE, :, :]
		_, loss_val = sess.run([train_op, loss], {x_placeholder:x_feed, y_placeholder:y_feed})

		batch_time += BATCH_SIZE






if __name__ == "__main__":
	x, y = generate_date()
	train(x, y)