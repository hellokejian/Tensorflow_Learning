import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
def function1():
    ops.reset_default_graph()
    x_batch = np.concatenate((np.random.normal(loc=-1, scale=1, size=50), np.random.normal(loc=3, scale=1, size=50)))
    y_batch = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))

    X = tf.placeholder(dtype=tf.float32, shape=[1])
    y_target = tf.placeholder(dtype=tf.float32, shape=[1])

    A = tf.Variable(tf.random_normal(shape=[1], mean=10))
    output = tf.add(X, A)
    output_expand = tf.expand_dims(output, 0)
    y_target_expand = tf.expand_dims(y_target, 0)

    x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=output_expand, labels=y_target_expand)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
    train_op = optimizer.minimize(x_entropy)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(1400):
            rand_index = np.random.choice(100)
            rand_x = [x_batch[rand_index]]
            rand_y = [y_batch[rand_index]]
            # sess.run(train_op, feed_dict={X:rand_x, y_train_target:rand_y})
            sess.run(train_op, feed_dict={X: rand_x, y_target: rand_y})
            if epoch % 5 == 0:
                A_val, loss = sess.run([A, x_entropy], feed_dict={X: rand_x, y_target: rand_y})
                print('epoch:', epoch, '\tA:', A_val, '\tx_entropy', loss)

def function2():
    iris = datasets.load_iris()
    iris_2d = np.array([[x[2], x[3]] for x in iris.data])
    binary_target = np.array([1.0 if x == 0 else 0. for x in iris.target])

    x_data = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    W_x = tf.Variable()

    return

def gate2():
    a = tf.Variable(tf.constant(5.))
    b = tf.Variable(tf.constant(1.))
    x_val = 5
    y_target = 100
    x_data = tf.placeholder(dtype=tf.float32,shape=[]) # , shape=[1]
    output = tf.add(tf.multiply(a, x_data), b)
    loss = tf.square(output - y_target)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(200):
            sess.run(train_op, feed_dict={x_data:x_val})
            if epoch % 5 == 0:
                loss_val, a_val, b_val = sess.run([loss, a, b], feed_dict={x_data:x_val})
                print('loss:', loss_val, '\ta:', a_val, '\tb', b_val)

def Combining_Gates_and_Activation_Functions():
    tf.set_random_seed(5)
    np.random.seed(42)
    batch_size = 50
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    a1 = tf.Variable(tf.random_normal(shape=[1, 1]))
    b1 = tf.Variable(tf.random_uniform(shape=[1, 1]))
    a2 = tf.Variable(tf.random_normal(shape=[1, 1]))
    b2 = tf.Variable(tf.random_uniform(shape=[1, 1]))

    x_val = np.random.normal(2, 0.1, 500)
    # tf.expand_dims(x_val, axis=1)
    x_data = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    sigmoid_activation = tf.sigmoid(tf.add(tf.matmul(x_data ,a1), b1))
    relu_activation = tf.nn.relu(tf.add(tf.matmul(x_data, a2), b2))

    loss_1 = tf.reduce_mean(tf.square(sigmoid_activation - 0.75))
    loss_2 = tf.reduce_mean(tf.square(relu_activation - 0.75))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op_1 = optimizer.minimize(loss_1)
    train_op_2 = optimizer.minimize(loss_2)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    # config = tf.ConfigProto(gpu_options=gpu_options)
    sess.run(init)
    for epoch in range(50):
        for iteration in range(500 // batch_size):
            x_batch = np.expand_dims(x_val[iteration * batch_size:(iteration + 1) * batch_size], axis=1)
            sess.run(train_op_1, feed_dict={x_data: x_batch})
            sess.run(train_op_2, feed_dict={x_data: x_batch})
        # if epoch % 5 == 0:
        # loss_val_1 = sess.run(loss_1, feed_dict={x_data:x_batch})
        # loss_val_2 = sess.run(loss_2, feed_dict={x_data:x_batch})
        # print('epoch:', epoch, '\tloss1:', loss_val_1, '\tloss2:', loss_val_2)
    sess.close()

def one_layer_network():
    ops.reset_default_graph()
    iris = datasets.load_iris()
    x_vals = np.array([x[0:3] for x in iris.data])
    y_vals = np.array([x[3] for x in iris.data])
    train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
    sess = tf.Session()

def implementing_different_layers():
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import csv
    import os
    import random
    import numpy as np
    import random
    from tensorflow.python.framework import ops
    ops.reset_default_graph()
    sess = tf.Session()

    data_size = 25
    con_size = 5
    maxpool_size = 5
    stride_size = 1
    seed = 3
    np.random.seed(seed=seed)
    tf.set_random_seed(seed=seed)
    data_1d = np.random.normal(size = data_size)

    x_input_1d = tf.placeholder(dtype=tf.float32, shape=[data_size])

    return

implementing_different_layers()
# one_layer_network()
# Combining_Gates_and_Activation_Functions()
# function2()
# gate2()