import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.python.framework import ops
ops.reset_default_graph()

data_dir = 'temp'
mnist = read_data_sets(data_dir)

train_images        = np.array([np.reshape(x, (28, 28)) for x in mnist.train.images])
test_images         = np.array([np.reshape(x, (28, 28)) for x in mnist.test.images])
train_labels        = mnist.train.labels
test_labels         = mnist.test.labels
image_width         = train_images[0].shape[0]
image_height        = train_images[0].shape[1]

batch_size          = 100
eval_every          = 5
learning_rate       = 0.05
n_epoch             = 1000
test_size           = 500

conv1_features      = 25
conv2_features      = 50
num_channels        = 1
max_pooling_size1   = 2
max_pooling_size2   = 2
fc1_size            = 100
target_size         = max(train_labels) + 1

# 输入卷积层的格式为[batch_size, image_width, image_height, num_channels]
train_input_shape   = [batch_size, image_width, image_height, num_channels]
x_train_input       = tf.placeholder(dtype=tf.float32, shape=train_input_shape)
y_train_target      = tf.placeholder(dtype=tf.int32, shape=(batch_size))

test_input_shape    = [test_size, image_width, image_height, num_channels]
x_test_input        = tf.placeholder(dtype=tf.float32, shape=test_input_shape)
y_test_target       = tf.placeholder(dtype=tf.int32, shape=(test_size))

# ------------------------------------------卷积层------------------------------------------

conv1_weight = tf.Variable(tf.truncated_normal(shape=[4, 4, num_channels, conv1_features]
                                               , stddev=0.1, dtype=tf.float32))
conv1_bias = tf.Variable(tf.zeros(shape=[conv1_features], dtype=tf.float32))
conv2_weight = tf.Variable(tf.truncated_normal(shape=[4, 4, conv1_features, conv2_features]
                                               , stddev=0.1, dtype=tf.float32))
conv2_bias = tf.Variable(tf.zeros(shape=[conv2_features], dtype=tf.float32))


def my_conv2d(input_data, conv_weight, conv_bias,strides = 1, padding = 'SAME', max_pooling_size = 2):
    conv = tf.nn.conv2d(input=input_data, filter=conv_weight
                        , strides=[1, strides, strides, 1], padding=padding)
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv_bias))
    max_pool = tf.nn.max_pool(value=relu, ksize=[1, max_pooling_size, max_pooling_size, 1]
                              , strides=[1, max_pooling_size, max_pooling_size, 1], padding=padding)
    return max_pool

def my_model(input_data):
    # ------------------------------------------卷积层------------------------------------------
    # conv层的weight格式为[filter_width, filter_height, channels, conv_features]
    conv1_weight = tf.Variable(tf.truncated_normal
                               (shape=[4, 4, num_channels, conv1_features], stddev=0.1, dtype=tf.float32))
    conv1_bias = tf.Variable(tf.zeros(shape=[conv1_features], dtype=tf.float32))
    conv2_weight = tf.Variable(tf.truncated_normal(shape=[4, 4, conv1_features, conv2_features], stddev=0.1, dtype=tf.float32))
    conv2_bias = tf.Variable(tf.zeros(shape=[conv2_features], dtype=tf.float32))

    conv_layer1 = my_conv2d(input_data, conv1_weight, conv1_bias, 1, 'SAME', max_pooling_size1)
    layer2 = my_conv2d(conv_layer1, conv2_weight, conv2_bias, 1, 'SAME', max_pooling_size2)

    # 卷积层的输出格式为batch_size, width, height, feature_map_no
    layer2_output_shape = layer2.get_shape().as_list()
    # 输入层单元的个数
    layer2_final_shape = layer2_output_shape[1] * layer2_output_shape[2] * layer2_output_shape[3]
    # 将输入调整为batch_size * 输入单元个数格式的矩阵
    layer2_final = tf.reshape(layer2, [layer2_output_shape[0], layer2_final_shape])

    layer2_output_width = image_width // (max_pooling_size1 * max_pooling_size2)
    layer2_output_height = image_height // (max_pooling_size1 * max_pooling_size2)

    layer2_output_size = layer2_output_height * layer2_output_width * conv2_features

    # -----------------------------------------------全连接层-----------------------------------------------
    fc1_weight = tf.Variable(tf.truncated_normal(shape=[layer2_output_size, fc1_size], stddev=0.1, dtype=tf.float32))
    fc1_bias = tf.Variable(tf.truncated_normal(shape=[fc1_size], stddev=0.1, dtype=tf.float32))
    fc2_weight = tf.Variable(tf.truncated_normal(shape=[fc1_size, target_size], stddev=0.1, dtype=tf.float32))
    fc2_bias = tf.Variable(tf.truncated_normal(shape=[target_size], stddev=0.1, dtype=tf.float32))

    layer3 = tf.nn.relu(tf.add(tf.matmul(layer2_final, fc1_weight), fc1_bias))
    # layer4 = tf.nn.relu(tf.matmul(layer3, fc2_weight) + fc2_bias)
    layer4 = tf.add(tf.matmul(layer3, fc2_weight),fc2_bias)
    return layer4

train_model_output = my_model(x_train_input)
test_model_output = my_model(x_test_input)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train_target, logits=train_model_output))

prediction = tf.nn.softmax(train_model_output)
test_prediction = tf.nn.softmax(test_model_output)

def get_accuracy(logits, targets):
    batch_prediction = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(batch_prediction, targets))
    return (100. * num_correct / batch_prediction.shape[0])

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

train_loss = []
train_acc = []
test_acc = []

for epoch in range(n_epoch):
    rand_index = np.random.choice(len(train_images), size=batch_size)
    rand_x = train_images[rand_index]
    rand_x = np.expand_dims(rand_x, 3)
    rand_y = train_labels[rand_index]
    train_dict = {x_train_input:rand_x, y_train_target:rand_y}
    sess.run(train_op, feed_dict=train_dict)
    temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict)
    temp_train_acc = get_accuracy(logits=temp_train_preds, targets=rand_y)

    if (epoch + 1) % eval_every == 0:
        eval_index = np.random.choice(len(test_images), size=test_size)
        eval_x = test_images[eval_index]
        eval_x = np.expand_dims(eval_x, 3)
        eval_y = test_labels[eval_index]
        test_dict = {x_test_input:eval_x, y_test_target:eval_y}
        temp_test_prediction = sess.run(test_prediction, feed_dict=test_dict)
        temp_test_acc = get_accuracy(logits=temp_test_prediction, targets=eval_y)

        train_loss.append(temp_train_loss)
        train_acc.append(temp_train_acc)
        test_acc.append(temp_test_acc)
        acc_loss = [(epoch + 1), temp_train_loss, temp_train_acc, temp_test_acc]
        acc_and_loss = [np.round(x, 2) for x in acc_loss]

        print('epoch # {}.Train Loss: {:.2f}. Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))

# Matlotlib code to plot the loss and accuracies
eval_indices = range(0, n_epoch, eval_every)
# Plot loss over time
plt.plot(eval_indices, train_loss, 'k-')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
plt.show()

# Plot train and test accuracy
plt.plot(eval_indices, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(eval_indices, test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

sess.close()

