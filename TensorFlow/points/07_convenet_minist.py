import os
import time
import TensorFlow as tf
import TensorFlow.utilss as utils
import TensorFlow.contrib.layers as layers
from TensorFlow.examples.tutorials.mnist import input_data
import TensorFlow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
N_CLASSES = 10

# step1: reading in data
# using tf learn's built in function to load minist data to the folder data/minist
mnist = input_data.read_data_sets("/data/minist", one_hot=True)
# step2: define parameters for model
LEARNING_RATE = 0.001
BATCH_SIZE = 128
SKIP_STEP = 10
DROPOUT = 0.75
N_EPOCHS = 1

# step3: create placeholders for features and labels
with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, shape=[None, 784], name='X_placeholder')
    Y = tf.placeholder(tf.float32, shape=[None, 10], name='Y_placeholder')
dropout = tf.placeholder(tf.float32, name='dropout')

# step4,5: create weights, do inference
# define the model struct:conv-relu-pool-conv-relu-pool-fully-softmax
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

with tf.variable_scope('conv1') as scope:
    # first,reshape the image to [BATCH_SIZE, 28, 28, 1] to make to work with tf.DeepLearning.conv2d
    images = tf.reshape(name='images', tensor=X, shape=[-1, 28, 28, 1])
    kernel = tf.get_variable(name='kernel', dtype=tf.float32, shape=[5, 5, 1, 32], initializer=tf.truncated_normal_initializer())
    biases = tf.get_variable(name='biases',dtype=tf.float32, shape=[32],initializer=tf.random_normal_initializer() )
    conv = tf.nn.conv2d(images, kernel, strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.relu(conv + biases, name=scope.name)

    # out put is of dimension batch_size28*28*32
    conv1 = layers.conv2d(images, 32, 5, 1, activation_fn=tf.nn.relu, padding='SAME')

with tf.variable_scope('pool1') as scope:
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # output is of dimension BATCH_SIZE x 14 x 14 x 32

with tf.variable_scope('conv2') as scope:
    # similar to conv1, except kernel now is of the size 5 x 5 x 32 x 64
    kernel = tf.get_variable(name='kernel', shape=[5,5,32,64], dtype=tf.float32,initializer=tf.truncated_normal_initializer())
    biases = tf.get_variable(name='biases', shape=[64],initializer=tf.random_normal_initializer())
    conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv + biases, name=scope.name)
    # output is of dimension BATCH_SIZE x 14 x 14 x 64
    # layers.conv2d(images, 64, 5, 1, activation_fn=tf.DeepLearning.relu, padding='SAME')

with tf.variable_scope('pool2') as scope:
    # similar to pool1
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # output is of dimension BATCH_SIZE x 7 x 7 x 64

with tf.variable_scope('fc') as scope:
    # use weight of dimension 7 * 7 * 64 x 1024
    input_features = 7 * 7 * 64
    w = tf.get_variable('weights', [input_features, 1024], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', [1024], initializer=tf.constant_initializer(0.0))

    # reshape pool2 to 2 dimensional
    pool2 = tf.reshape(pool2, [-1, input_features])
    fc = tf.nn.relu(tf.matmul(pool2, w) + b, name='relu')
    # pool2 = layers.flatten(pool2)
    # fc = layers.fully_connected(pool2, 1024, tf.DeepLearning.relu)
    fc = tf.nn.dropout(fc, dropout, name='relu_dropout')
with tf.variable_scope('softmax_linear') as scope:
    w = tf.get_variable('weights', [1024, N_CLASSES], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', [N_CLASSES], initializer=tf.random_normal_initializer())
    logits = tf.matmul(fc, w) + b

# Step 6: define loss function

# use softmax cross entropy with logits as the loss function

# compute mean cross entropy, softmax is applied internally

with tf.name_scope('loss'):
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
    loss = tf.reduce_mean(entropy, name='loss')

with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('histogram loss', loss)
    summary_op = tf.summary.merge_all()

# Step 7: define training op

# using gradient descent with learning rate of LEARNING_RATE to minimize cost

optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)
utils.make_dir('checkpoints')
utils.make_dir('checkpoints/convnet_mnist')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # to visualize using TensorBoard
    writer = tf.summary.FileWriter('./graphs/convnet', sess.graph)
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_mnist/checkpoint'))
    # if that checkpoint exists, restore from checkpoint
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    initial_step = global_step.eval()
    start_time = time.time()
    n_batches = int(mnist.train.num_examples / BATCH_SIZE)
    total_loss = 0.0

    for index in range(initial_step, n_batches * N_EPOCHS):  # train the model n_epochs times
        X_batch, Y_batch = mnist.train.next_batch(BATCH_SIZE)
        _, loss_batch, summary = sess.run([optimizer, loss, summary_op], feed_dict={X: X_batch, Y: Y_batch, dropout: DROPOUT})
        writer.add_summary(summary, global_step=index)
        total_loss += loss_batch
        if (index + 1) % SKIP_STEP == 0:
            print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss / SKIP_STEP))
            total_loss = 0.0
            saver.save(sess, 'checkpoints/convnet_mnist/mnist-convnet', index)
    print("Optimization Finished!")  # should be around 0.35 after 25 epochs
    print("Total time: {0} seconds".format(time.time() - start_time))

    # test the model
    n_batches = int(mnist.test.num_examples / BATCH_SIZE)
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(BATCH_SIZE)
        _, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict={X: X_batch, Y: Y_batch, dropout: 1.0})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += sess.run(accuracy)
    print("Accuracy {0}".format(total_correct_preds / mnist.test.num_examples))




