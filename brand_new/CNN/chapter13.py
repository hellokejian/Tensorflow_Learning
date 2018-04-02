import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_images
from matplotlib.image import imread
# config = tf.ConfigProto()
def cnn1():
    # load data simples
    dataset = np.array(load_sample_images().images, dtype=np.float32)
    batch_size, height, width, channels = dataset.shape

    # create two filters
    filter_test = np.zeros((7, 7, channels, 2), dtype=np.float32)
    filter_test[:, 3, :, 0] = 1 #vertical line
    filter_test[3, :, :, 1] = 1 #horizontal line

    # create a graph with input x + a conv layer applying 2 filters
    x = tf.placeholder(tf.float32, shape=[None, height, width, channels])
    convolution = tf.nn.conv2d(x, filter_test, strides=[1, 2, 2, 1], padding='SAME')
    with tf.Session() as sess:
        output = sess.run(convolution, feed_dict={x:dataset})
    plt.imshow(output[0, :, :, 0])
    plt.imshow(output[0, :, :, 1])
    plt.show()
    return 0

def cnn_with_maxpooling():
    # load data simples
    dataset = np.array(load_sample_images().images, dtype=np.float32)
    batch_size, height, width, channels = dataset.shape

    # create two filters
    filter_test = np.zeros((7, 7, channels, 2), dtype=np.float32)
    filter_test[:, 3, :, 0] = 1  # vertical line
    filter_test[3, :, :, 1] = 1  # horizontal line

    # create a graph with input x + a conv layer applying 2 filters
    x = tf.placeholder(tf.float32, shape=[None, height, width, channels])
    # convolution = tf.nn.conv2d(x, filter_test, strides=[1, 2, 2, 1], padding='SAME')
    max_pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    with tf.Session() as sess:
        output = sess.run(max_pool, feed_dict={x: dataset})
    # plt.imshow(output[0, :, :, 0])
    plt.imshow(output[0].astype(np.uint8))
    plt.show()
    return 0

def Inception_v3():
    print('pain in the ass')

    print('done')
    return

# cnn1()
# cnn_with_maxpooling()
Inception_v3()