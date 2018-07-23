import tensorflow as tf

EMA_DECAY = 0.99
epoch = 100

ema_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
ema = tf.train.ExponentialMovingAverage(decay=EMA_DECAY, num_updates=ema_step)

v = tf.Variable(100000.0, name="v", dtype=tf.float32)
# 维护影子变量，计算并保存变量的平均值，ema.average(v)返回平均值
v_ema_op = ema.apply([v])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(v_ema_op)
    print("开始值：{0}".format(sess.run([v, ema.average(v)])))
    sess.run(tf.assign(v, 10000.0))
    for i in range(epoch):
        # sess.run(tf.assign_add(v, 100))
        print("v={0}, num_step={1},decay={2}".format(sess.run(v), sess.run(ema_step),
              sess.run(tf.minimum(EMA_DECAY, tf.cast((1+ema_step)/(10+ema_step), tf.float32)))))
        sess.run(v_ema_op)
        print(sess.run([v, ema.average(v)]))
        sess.run(tf.assign_add(ema_step,1))
