import cifar10, cifar10_input
import  tensorflow as tf
import numpy as np
import time

data_dir = 'D:\workspace\python\datasets\cifar10\cifar-10-batches-bin'
max_steps = 3000
batch_size = 128
def variable_with_weighted_loss(shape, stddev, w_loss):
    var = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32))
    if w_loss is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w_loss, name='w_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

# cifar10.maybe_download_and_extract()

images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

'''conv layer1'''
conv1_weight = variable_with_weighted_loss(shape=[5, 5, 3, 64], stddev=0.05, w_loss=0)
conv1_kernel = tf.nn.conv2d(image_holder, conv1_weight, [1, 1, 1, 1], padding='SAME')

conv1_bias = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32))

conv1_add_bias = tf.nn.relu(tf.nn.bias_add(conv1_kernel, conv1_bias))

max_pool1 = tf.nn.max_pool(conv1_add_bias, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

norm1 = tf.nn.lrn(max_pool1, bias=1.0, alpha=0.001 / 9., beta=0.75)

conv1 = norm1

'''conv layer2'''
conv2_weight = variable_with_weighted_loss(shape=[5, 5, 64, 64], stddev=0.5e-2, w_loss=0)
conv2_kernel = tf.nn.conv2d(norm1, conv2_weight, [1, 1, 1, 1], padding='SAME')

conv2_bias = tf.Variable(tf.constant(value=0.1 ,shape=[64], dtype=tf.float32))

conv2_add_bias = tf.nn.relu(tf.nn.bias_add(conv2_kernel, conv2_bias))

norm2 = tf.nn.lrn(conv2_add_bias, bias=1.0, alpha=0.001 / 9., beta=0.75)

max_pool2 = tf.nn.max_pool(norm2, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
conv2 = max_pool2

'''fully connected layer1'''
print(conv2.get_shape())
conv2_output_flatten = tf.reshape(conv2, shape=[batch_size, -1], name='conv2_output_flatten')
print(conv2_output_flatten.get_shape()[0].value)

fc1_input = conv2_output_flatten.get_shape()[1].value
fc1_output = 384
fc1_weight = variable_with_weighted_loss(shape=[fc1_input, fc1_output], stddev=0.04, w_loss=0.004)
fc1_bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[fc1_output]))

fc1_add_bias = tf.nn.bias_add(tf.matmul(conv2_output_flatten, fc1_weight), fc1_bias)

fc1 = tf.nn.relu(fc1_add_bias)

'''fully connected layer2'''
fc2_input = 384
fc2_output = 192
fc2_weight = variable_with_weighted_loss(shape=[fc2_input, fc2_output], stddev=0.04, w_loss=0.004)
fc2_bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[fc2_output]))

fc2_add_bias = tf.nn.bias_add(tf.matmul(fc1, fc2_weight), fc2_bias)

fc2 = tf.nn.relu(fc2_add_bias)

'''fully connected layer3'''
fc3_input = 192
fc3_output = 10
fc3_weight = variable_with_weighted_loss(shape=[fc3_input, fc3_output], stddev=1 / 192, w_loss=0.)
fc3_bias = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[fc3_output]))

fc3_add_bias = tf.nn.bias_add(tf.matmul(fc2, fc3_weight), fc3_bias)

logits = fc3_add_bias

def get_loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    # print(cross_entropy.get_shape())
    # print(labels.get_shape())
    loss = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('loss', loss)
    return tf.add_n(tf.get_collection('loss'), name='total_loss')

loss = get_loss(logits, label_holder)

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

top_k = tf.nn.in_top_k(logits, label_holder, 1)
accuracy_op = tf.reduce_mean(tf.cast(top_k, tf.float32))

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

tf.train.start_queue_runners()

for epoch in range(max_steps):
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss],
                       feed_dict={image_holder:image_batch, label_holder:label_batch})
    print('epoch', epoch + 1, 'loss', loss_value)

sess.close()