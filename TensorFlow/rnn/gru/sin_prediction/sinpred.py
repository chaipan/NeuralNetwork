import tensorflow as tf
import numpy as np
import tensorflow.contrib.rnn as rnn
from TensorFlow.rnn.gru.sin_prediction.generate_sample import generate_sample
# from TensorFlow.rnn.gru.sin_prediction import


LEARNINT_RATE = 0.001
EPOCHES = 10000
BATCH_SIZE = 50

INPUT_SIZE = 1
SERIES_LENGTH = 25
HIDDEN_NUMS = 32
OUT_PUTS = 100
GRU_CELLS = 4

x = tf.placeholder(dtype=tf.float32, shape=[None, SERIES_LENGTH, INPUT_SIZE])
y = tf.placeholder(dtype=tf.float32, shape=[None, OUT_PUTS])

gru_cells = [rnn.GRUCell(HIDDEN_NUMS) for _ in range(GRU_CELLS)]
stacked_gru = rnn.MultiRNNCell(gru_cells)
gru_out_puts, states = tf.nn.dynamic_rnn(stacked_gru,
									 inputs=x,
									 dtype=tf.float32)
pred = tf.layers.dense(gru_out_puts[:,-1, :],
						   units=OUT_PUTS)
error = tf.squared_difference(y, pred)
# losses = tf.reduce_sum(error, axis=1)
loss = tf.reduce_mean(error)

train_op = tf.train.AdamOptimizer(LEARNINT_RATE).minimize(loss)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

sess = tf.Session()
sess.run(init_op)

for step in range(EPOCHES):
	_, batch_x, __, batch_y = generate_sample(f=None, t0=None, batch_size=BATCH_SIZE, samples=SERIES_LENGTH,
											  predict=OUT_PUTS)
	batch_x = np.reshape(batch_x, [BATCH_SIZE, SERIES_LENGTH, INPUT_SIZE])
	batch_y = np.reshape(batch_y, [BATCH_SIZE, OUT_PUTS])

	_, loss_val = sess.run([train_op, loss], {x:batch_x, y:batch_y})
	print(loss_val)





