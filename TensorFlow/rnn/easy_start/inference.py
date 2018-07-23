import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt


"""
    生成训练数据,假设每个输入维度为1*1
"""


STATE_SIZE = 4
NUM_CLASSES = 2


def generate_data(data_size=7500,series_length=15):
    # 从[0,1]中生成7500个数，0，1的出现概率分别是0.5，0.5
    x = np.random.choice([0,1], size=data_size, p=[0.5,0.5])
    # 把x的元素右移三个生成y
    y = np.roll(x, shift=3)

    x_series = np.reshape(x, newshape=[-1, series_length]) # 500行15列
    y_series = np.reshape(y, newshape=[-1, series_length]) # 500行15列

    print("x.shape", x_series.shape, "y.shape", y_series.shape)
    return x_series, y_series

"""
源程序中对于训练数据的处理比较复杂，处理过程实现的功能是将500*15的数据分成100组，每组5*15，每组存成一个训练序列，每个单次输入去该组的每一列，以后往后，
当一组取完后取下一组
"""

def inference(batch_x_series, batch_size):
    w = tf.Variable(initial_value=np.random.rand(STATE_SIZE + 1, STATE_SIZE), dtype=tf.float32)
    b = tf.Variable(initial_value=np.random.random(STATE_SIZE), dtype=tf.float32)
    w2 = tf.Variable(initial_value=np.random.rand(STATE_SIZE, NUM_CLASSES), dtype=tf.float32)
    b2 = tf.Variable(initial_value=np.random.rand(NUM_CLASSES), dtype=tf.float32)
    init_state = np.zeros((batch_size, STATE_SIZE))

    # 输入序列，长度为15，分割成每个单独的输入，每个大小为1，使用batch输入，每个为batch*1大小
    # 对矩阵沿着列方向做一次stack相当于做了一次转置，由于转置没有方向性，所以stack=unstack
    input_series = tf.unstack(batch_x_series, axis=1)
    state_series = []
    current_state = init_state




    # 一次计算出一个序列的输出
    for current_input in input_series:
        # 此时的current_input形状为（5，）
        current_input = tf.reshape(current_input, shape=[batch_size, 1])
        # 将当前输入和当前累计状态进行拼接，1代表沿着列合并
        input_and_concatenated = tf.concat([current_input, current_state], 1)
        next_state = tf.tanh(tf.matmul(input_and_concatenated, w) + b)
        # 将循环体输出按时间顺序存起来
        state_series.append(next_state)
        current_state = next_state

    return state_series

def plot(loss_list, batch_x, batch_y):
    plt.subplot(2,3,1)
    plt.plot(loss_list)
    for batch_series_idx in range(5):
        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, 15, 0, 2])
        left_offset = range(15)
        plt.bar(left_offset, batch_x[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batch_y[batch_series_idx, :] * 0.5, width=1, color="red")

        plt.draw()
        plt.pause(0.0001)






if __name__ == "__main__":
    pass