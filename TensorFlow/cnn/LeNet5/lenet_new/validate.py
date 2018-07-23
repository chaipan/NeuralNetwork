import tensorflow as tf
import numpy as np
# from TensorFlow.cnn.LeNet5.lenet_new import inference as inference

"""
每次取1000条数据，取五次，平均计算正确率
"""
def validate(data, inference):
	validation = data.validation
	accuracy_list = []
	for i in range(5):
		# np.random.shuffle(validation)
		x, y = validation.next_batch(1000)
		x_reshaped = np.reshape(x, [1000, 28, 28, 1])
		logits = inference.inference(x_reshaped)
		accucary = np.mean(np.equal(np.argmax(y, 1), np.argmax(logits, 1)))
		accuracy_list.append(accucary)
		print("数据验证， step{0}， 准确率{1:.5f}".format(i, accucary))


	return np.mean(accuracy_list)




def test():
	a = [1,2,3]
	b = [1,0, 2]
	print(np.equal(a, b))


if __name__ == "__main__":
	test()