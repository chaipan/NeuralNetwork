import TensorFlow as tf
from TensorFlow.contrib import layers

def cross_entropy_with_softMax():

    tf.nn.sparse_softmax_cross_entropy_with_logits()
    layers.l2_regularizer()