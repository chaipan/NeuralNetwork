import TensorFlow as tf

"""
    模型加载中对滑动平均方法的应用
"""

def saver_ema_test(arg=None):
    v = tf.Variable(0.0, name="v")
    for variables in tf.global_variables():
        print(variables.name)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    ema_op = ema.apply(tf.global_variables())
    for variables in tf.global_variables():
        print(variables.name)
    """在滑动平均模型中，会为每个变量维护一个影子变量，所以上面输出v/ExponentialMovingAverage:0"""

    writer = tf.summary.FileWriter("graph/", graph=tf.get_default_graph())
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(v, 10.0))
        sess.run(ema_op)
        writer.flush()

        print(sess.run(ema.average(v)))
        # 此处模型会将影子变量也保存在model中
        saver.save(sess, "model/saver_ema.ckpt")



def restore_v(arg=None):
    v = tf.Variable(0.0, name="v")
    saver = tf.train.Saver({"v/ExponentialMovingAverage":v})
    with tf.Session() as sess:
        saver.restore(sess, "model/saver_ema.ckpt")
        for varibales in tf.global_variables():
            print(varibales.name)
        print(sess.run(v))

def variables_to_store_test(arg=None):
    v1 = tf.Variable(0.1, name="v")
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    # variables_to_restore()可将ema可维护的变量影子变量全部列出来
    ema_variables_to_store = ema.variables_to_restore()

    print(ema_variables_to_store)
    # 自动将影子变量的值加载到当前变量，对应规则为影子变量的name："v/ExponentialMovingAverage"前面的v与当前计算图的变量名v相等v1 = tf.Variable(0.1, name="v")
    saver = tf.train.Saver(ema_variables_to_store)
    with tf.Session() as sess:
        saver.restore(sess, "model/saver_ema.ckpt") #此处不要用sess.run()
        print(sess.run(v1))





def main(arg=None):
    # saver_ema_test()
    # restore_v()
    variables_to_store_test()

if __name__ == "__main__":
    main()