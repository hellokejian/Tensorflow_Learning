import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
def weight_variable(shape):
    initial_weight = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial_weight)
def bias_variable(shape):
    initial_bias = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_bias)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_Pool_2v2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

batch_size = 50
batch_num = mnist.train.num_examples // batch_size

# 定义卷积层
weight_1 = weight_variable([5, 5, 1, 32])
bias_1 = bias_variable([32])
conv2d_1 = tf.nn.relu(conv2d(x_image, weight_1) + bias_1)
pool2d_1 = max_Pool_2v2(conv2d_1)

#定义第二个卷积层
weight_2 = weight_variable([5, 5, 32, 64])
bias_2 = bias_variable([64])
con2d_2 = tf.nn.relu(conv2d(pool2d_1, weight_2) + bias_2)
pool2d_2 = max_Pool_2v2(con2d_2)

# 转入全连接层
weight_fc1 = weight_variable([49 * 64, 1024])
bias_fc1 = bias_variable([1024])
pool2d_2_flat = tf.reshape(pool2d_2 ,[-1, 7*7*64])
fc_1 = tf.nn.relu(tf.matmul(pool2d_2_flat, weight_fc1) + bias_fc1)

# 连接一个dropout正则化层防止过拟合
keep_prob = tf.placeholder(tf.float32)
fc_1_drop = tf.nn.dropout(fc_1, keep_prob=keep_prob)

# 接入一个softmax层，得到最后的概率输出
weight_fc2 = weight_variable([1024, 10])
bias_fc2 = bias_variable([10])
fc_2 = tf.nn.softmax(tf.matmul(fc_1_drop, weight_fc2) + bias_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(fc_2), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)

correction_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(fc_2, 1))
accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoh in range(2000):
        for batch in range(batch_num):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys, keep_prob:0.8})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        if (epoh % 100 == 0):
            print('epoh:' + str(epoh), 'accuracy:'+str(acc))
