import numpy as np

"""
非向量化的svm损失函数计算，使用cifar10数据集，
x为列向量，y真实标签值得索引，w为10*x.shape[0]的矩阵
x,y都是用增加偏置的预处理方式
score function使用线性函数
"""
def L_i(x,y,w,delta):
    num_class = w.shape[0]#类别数
    scores = w.dot(x)#
    s_y = scores[y,0]
    loss = 0
    for i in range(num_class):
        if i == y:
            continue
        loss += max(0, scores[i,0] - s_y + delta)
    return loss


def L_i_vector(x, y, w, delta):
    scores = w.dot(x)
    margins = np.maximum(0, scores - scores[y,0] + delta)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i