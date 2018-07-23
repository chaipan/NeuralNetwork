from __future__ import print_function,division
import tensorflow as tf
import matplotlib.pylab as plt
from TensorFlow.rnn.easy_start import inference

"""
    循环神经网络启蒙案例，copy自Erik Hallström
"""

DATA_SIZE = 7500
SERIES_LENGTH = 15
EPOCHES = 100
BATCH_SIZE = 5
NUM_BATCHES = DATA_SIZE//BATCH_SIZE//SERIES_LENGTH

"""构建计算图"""
batchX_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SERIES_LENGTH])
batchY_placeholder = tf.placeholder(tf.int32, shape=[BATCH_SIZE, SERIES_LENGTH])

# 输入序列，长度为15，分割成每个单独的输入，每个大小为1，使用batch输入，每个为batch*1大小
labels_series = tf.unstack(batchY_placeholder, axis=1)
state_series = inference.inference(batch_x_series=batchX_placeholder, batch_size=5)
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=state_series,labels=labels_series)
total_loss = tf.reduce_mean(losses)


train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(EPOCHES):
        x,y = inference.generate_data()
        print("New data, epoch {}", epoch_idx)
        for batch_idx in range(NUM_BATCHES):
            # 构造每个输入序列的数据下标
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            # 将生成的数据分割成每个长度为15的输入序列，没考虑右边界，python右边界越界会自动收缩到右边界
            batchX = x[start_idx:end_idx, :]
            batchY = y[start_idx:end_idx, :]

            _total_loss, _train_step = sess.run(
                [total_loss, train_step],
                    feed_dict={
                        batchX_placeholder:batchX,
                        batchY_placeholder:batchY
                    })
            loss_list.append(_total_loss)
            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                inference.plot(loss_list, batchX, batchY)

    plt.ioff()
    plt.show()

def main(arg=None):
    pass
# if __name__ == "__main__":
#     main()



