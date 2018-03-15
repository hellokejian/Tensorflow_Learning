import tensorflow as tf
import cifar10, cifar10_input
import numpy as np
import time
max_steps = 2000
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'
def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection(name='losses', value=weight_loss)
    return var

# cifar10.maybe_download_and_extract()

images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

print('数据载入完毕')

images_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
labels_holder = tf.placeholder(tf.int32, [batch_size])

weight_1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
kernel_1 = tf.nn.conv2d(images_holder, weight_1, [1, 1, 1, 1], padding='SAME')
bias_1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv_1 = tf.nn.relu(tf.nn.bias_add(kernel_1, bias_1))
pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 3, 3, 1], strides=[1 ,2, 2, 1], padding='SAME')
norm_1 = tf.nn.lrn(pool_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

weight_2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
kernel_2 = tf.nn.conv2d(norm_1, weight_2, [1, 1, 1, 1], padding='SAME')
bias_2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv_2 = tf.nn.relu(tf.nn.bias_add(kernel_2, bias_2))
norm_2 = tf.nn.lrn(conv_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool_2 = tf.nn.max_pool(norm_2,ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

shape_1 = tf.reshape(pool_2, [batch_size, -1])
dim = shape_1.get_shape()[1].value
weight_3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
bias_3 = tf.Variable(tf.constant(0.1, shape=[384]))
local_3 = tf.nn.relu(tf.matmul(shape_1, weight_3) + bias_3)

weight_4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
bias_4 = tf.Variable(tf.constant(0.1, shape=[192]))
local_4 = tf.nn.relu(tf.matmul(local_3, weight_4) + bias_4)

weight_5 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, w1=0.0)
bias_5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local_4, weight_5), bias_5)

print('参数初始化完毕')

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total-loss')

loss = loss(logits, labels_holder)
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(logits, labels_holder, 1)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
tf.train.start_queue_runners()
for step in range(max_steps):
   start_time = time.time()
   images_batch, label_batch = sess.run([images_train, labels_train])
   _, loss_value = sess.run([train_step, loss], feed_dict={images_holder:images_batch, labels_holder:label_batch})
   duration = time.time() - start_time
   if step % 10 == 0:
       example_per_sec = batch_size / duration
       second_per_batch = float(duration)
       format_str = ('step %d,loss=%.2f (%.1f examples/second; %.3f sec/batch)')
       print(format_str % (step, loss_value, example_per_sec, second_per_batch))

num_examples = 10000
import math
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    images_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={images_holder:images_batch, labels_holder:label_batch})
    true_count += np.sum(predictions)
    step += 1
precision = true_count / total_sample_count
print('precision @ 1 = %3.f' % precision)
sess.close()
