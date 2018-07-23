"""

tensorflow 中 softmax_cross_entropy_with_logits 与 sparse_softmax_cross_entropy_with_logits 的区别

Having two different functions is a convenience, as they produce the same result.
The difference is simple:
For sparse_softmax_cross_entropy_with_logits, labels must have the shape [batch_size] and the dtype int32 or int64. Each label is an int in range [0, num_classes-1].
For softmax_cross_entropy_with_logits, labels must have the shape [batch_size, num_classes] and dtype float32 or float64.
Labels used in softmax_cross_entropy_with_logits are the one hot version of labels used in sparse_softmax_cross_entropy_with_logits.
Another tiny difference is that with sparse_softmax_cross_entropy_with_logits, you can give -1 as a label to have loss 0 on this label.
"""