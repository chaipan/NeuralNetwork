"""

交叉熵可在神经网络(机器学习)中作为损失函数，p表示真实标记的分布，q则为训练后的模型的预测标记分布，
交叉熵损失函数可以衡量p与q的相似性。交叉熵作为损失函数还有一个好处是使用sigmoid函数在梯度下降时能避免均方误差损失函数学习速率降低的问题，
因为学习速率可以被输出的误差所控制。tensorflow中自带的函数可以轻松的实现交叉熵的计算。
tf.nn.softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, dim=-1, name=None)
Computes softmax cross entropy between logits and labels.
注意：如果labels的每一行是one-hot表示，也就是只有一个地方为1，其他地方为0，可以使用tf.sparse_softmax_cross_entropy_with_logits()
警告：
1. 这个操作的输入logits是未经缩放的，该操作内部会对logits使用softmax操作
2. 参数labels,logits必须有相同的形状 [batch_size, num_classes] 和相同的类型(float16, float32, float64)中的一种


下面来看tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)
这个函数与上一个函数十分类似，唯一的区别在于labels.
注意：对于此操作，给定标签的概率被认为是排他的。labels的每一行为真实类别的索引
警告：
1. 这个操作的输入logits同样是是未经缩放的，该操作内部会对logits使用softmax操作
2. 参数logits的形状 [batch_size, num_classes] 和labels的形状[batch_size]
"""