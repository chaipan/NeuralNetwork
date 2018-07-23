import TensorFlow as tf
import os

def saver_test(arg=None):
    v1 = tf.Variable(0.1, name="v1")
    v2 = tf.Variable(1.1)
    result = tf.add(v1, v2, name="result")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(result)
        # 相对路径，当前文件夹下建立model文件夹，模型名称为model
        saver.save(sess, os.path.join("model/", "saver_test.ckpt") )


def load_test(arg=None):
    saver = tf.train.import_meta_graph("model/saver_test.ckpt.meta")

    with tf.Session() as sess:
        saver.restore(sess, "model/saver_test.ckpt")
        print(sess.run(tf.get_default_graph().get_tensor_by_name("result:0")))

#       将原来的变量加载到现在的变量
def trans_test(arg=None):
    v1 = tf.Variable(initial_value=0.0)
    saver = tf.train.Saver({"v1":v1})
    with tf.Session() as sess:
        saver.restore(sess, "model/saver_test.ckpt")
        print(sess.run(v1))



def main(arg=None):
    saver_test()
    trans_test()
if __name__ == "__main__":
    main()