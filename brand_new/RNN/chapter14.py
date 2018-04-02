import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
def Simple_RNN():
    n_inputs = 3
    n_neurons = 5
    x0 = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs])
    x1 = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs])

    w_x = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons]), dtype=tf.float32)
    w_y = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
    b = tf.Variable(tf.zeros(shape=[1, n_neurons], dtype=tf.float32))
    y0 = tf.tanh(tf.matmul(x0, w_x) + b)
    y1 = tf.tanh(tf.matmul(x1, w_x) + tf.matmul(y0, w_y) + b)

    init = tf.global_variables_initializer()

    x0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])
    x1_batch = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1], [1, 9, 8]])
    with tf.Session() as sess:
        sess.run(init)
        Y0_val, Y1_val = sess.run([y0, y1], feed_dict={x0:x0_batch, x1:x1_batch})
    print(Y1_val)
    return

def Static_RNN():
    n_inputs = 3
    n_neurons = 5
    x0 = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs])
    x1 = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs])
    basic_cell = rnn.BasicRNNCell(num_units=n_neurons)
    output_sequence, states = rnn.static_rnn(cell=basic_cell, inputs=[x0, x1], dtype=tf.float32)
    y0, y1 = output_sequence

    x0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])
    x1_batch = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1], [1, 9, 8]])
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        y0_val, y1_val = sess.run([y0, y1], feed_dict={x0: x0_batch, x1:x1_batch})
    print("----------------------------y0_val----------------------------")
    print(y0_val)
    print("----------------------------y1_val----------------------------")
    print(y1_val)
    # n_inputs = 3
    # n_neurons = 5
    # x0 = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs])
    # x1 = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs])
    # # W_x = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
    # # W_y = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
    # # b = tf.Variable(tf.zeros(shape=[1, n_neurons]))
    #
    # basic_cell = rnn.BasicRNNCell(num_units=n_neurons)
    # output_sequence, states = rnn.static_rnn(cell=basic_cell, inputs=[x0, x1], dtype=tf.float32)
    # y0, y1 = output_sequence
    #
    # x0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])
    # x1_batch = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1], [1, 9, 8]])
    #
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     y0_val, y1_val = sess.run([y0, y1], feed_dict={x0: x0_batch, x1: x1_batch})
    #     print(y0_val)
    #     print(y1_val)
    return

def Static_RNN2():
    n_inputs = 3
    n_neurons = 5
    n_steps = 2
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])
    X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))
    basic_cell = rnn.BasicRNNCell(num_units=n_neurons)
    output_seqs, states = rnn.static_rnn(cell=basic_cell, inputs=X_seqs, dtype=tf.float32)
    outputs = tf.transpose(tf.stack(output_seqs), perm=[1, 0, 2])
    x_inputs = [[[0, 1, 2], [9, 8, 7]],
                [[3, 4, 5], [6, 5, 4]],
                [[6, 7, 8], [3, 2, 1]],
                [[9, 0, 1], [1, 9, 8]]
               ]
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        outputs_val = sess.run(outputs, feed_dict={X:x_inputs})
        print(outputs_val)

def dynamic_RNN():
    n_inputs = 3
    n_neurons = 5
    n_steps = 2
    X_seqs = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])
    basic_cell = rnn.BasicRNNCell(num_units=n_neurons)
    output_seqs, states = tf.nn.dynamic_rnn(cell=basic_cell, inputs=X_seqs, dtype=tf.float32)
    # outputs = tf.transpose(tf.stack(output_seqs), perm=[1, 0, 2])
    x_inputs = np.array()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        outputs_val = sess.run(output_seqs, feed_dict={X_seqs: x_inputs})
        print(outputs_val)
    return

def dynamic_RNN_with_Length():
    n_inputs = 3
    n_neurons = 5
    n_steps = 2
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])
    seq_length = tf.placeholder(tf.int32, shape=[None])
    basic_cell = rnn.BasicRNNCell(num_units=n_neurons)
    out_seqs, states = tf.nn.dynamic_rnn(cell=basic_cell, inputs=X, dtype=tf.float32, sequence_length=seq_length)
    X_batchs = np.array([[[0, 1, 2], [9, 8, 7]], #instance 0
                        [[3, 4, 5], [0, 0, 0]],  #instance 1, padding with 0
                        [[6, 7, 8], [3, 2, 1]],  #instance 2
                        [[9, 0, 1], [1, 9, 8]]   #instance 3
                        ])
    seq_length_batch = np.array([2, 1, 2, 2])
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        output_val, states_val = sess.run([out_seqs, states], feed_dict={X:X_batchs, seq_length:seq_length_batch})
        print(output_val)
        print(states_val)
    return

# output
# [[[ 0.88057524 -0.81273335  0.80683768  0.11895712  0.71505582]
#   [-0.84263742 -0.9999997   0.99999976 -0.85630721  0.9999578 ]]
#
#  [[ 0.90709454 -0.99938488  0.99967021 -0.27332088  0.99629968]
#   [ 0.          0.          0.          0.          0.        ]]
#
#  [[ 0.92795062 -0.99999815  0.99999946 -0.59179348  0.99995863]
#   [-0.94004726 -0.92577773  0.79023135 -0.06023521  0.78303367]]
#
#  [[-0.99995464 -0.99542397  0.999668   -0.99972159  0.99976695]
#   [ 0.99992323 -0.99992442  0.9998402   0.99932426  0.99704176]]]

# [[-0.84263742 -0.9999997   0.99999976 -0.85630721  0.9999578 ] final state of each cell
#  [ 0.90709454 -0.99938488  0.99967021 -0.27332088  0.99629968]
#  [-0.94004726 -0.92577773  0.79023135 -0.06023521  0.78303367]
#  [ 0.99992323 -0.99992442  0.9998402   0.99932426  0.99704176]]

def RNN_with_MNIST():
    n_inputs   = 28
    n_steps    = 28
    n_neurons  = 150
    n_outputs  = 10

    mnist = input_data.read_data_sets('MNIST') #, one_hot=True
    X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
    Y_test = mnist.test.labels

    learning_rate = 0.001
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])
    y = tf.placeholder(dtype=tf.int32, shape=[None])
    basic_cell = rnn.BasicRNNCell(num_units=n_neurons)
    outputs, states = tf.nn.dynamic_rnn(cell=basic_cell, inputs=X, dtype=tf.float32)

    logits = fully_connected(inputs=states, num_outputs=n_outputs, activation_fn=None) #
    x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(x_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss=loss)

    correct = tf.nn.in_top_k(predictions=logits, targets=y, k=1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    init = tf.global_variables_initializer()

    n_epochs = 100
    batch_size = 150

    X_batch = 0
    Y_batch = 0
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, Y_batch = mnist.train.next_batch(batch_size=batch_size)
                X_batch = X_batch.reshape((-1, n_steps, n_inputs))
                sess.run(train_op, feed_dict={X:X_batch, y:Y_batch})
            accuracy_train = accuracy.eval(feed_dict={X:X_batch, y:Y_batch})
            accuracy_test = accuracy.eval(feed_dict = {X:X_test, y:Y_test})
            print('epoch:', epoch, '\taccuracy_train:', accuracy_train, '\taccuracy_test:', accuracy_test)
    return

def test():
    mnist = input_data.read_data_sets('MNIST')
    y = mnist.train.labels
    print(np.shape(y))

    a = np.array([1, 2, 3, 4, 5])
    print(a[-5:])

def output_projection_wrapper():
    n_inputs      = 1
    n_steps       = 20
    n_neurons     = 100
    n_outputs     = 1
    batch_size    = 50
    n_iteration   = 100000
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])

    basic_cell = rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    cell = rnn.OutputProjectionWrapper(cell=basic_cell, output_size=n_outputs)

    outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=X, dtype=tf.float32)
    learning_rate = 0.001
    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for iteration in range(n_iteration):
            # X_batch =
            return
    return

def tricker_RNN():
    n_inputs        = 28
    n_steps         = 28
    n_neurons       = 100
    n_outputs       = 10
    batch_size      = 50
    n_epoch         = 100
    learning_rate   = 0.001
    mnist = input_data.read_data_sets('MNIST')
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])
    y = tf.placeholder(dtype=tf.int32, shape=[None])

    basic_cell = rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    outputs, states = tf.nn.dynamic_rnn(cell=basic_cell, inputs=X, dtype=tf.float32)
    stacked_rnn_outputs = tf.reshape(outputs, shape=[-1, n_neurons])
    stacked_outputs = fully_connected(inputs=stacked_rnn_outputs, num_outputs=n_outputs, activation_fn=None)
    outputs = tf.reshape(stacked_outputs, shape=[-1, n_steps, n_outputs])
    logits = outputs[:, n_steps - 1, :]

    x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(x_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    correct = tf.nn.in_top_k(predictions=logits, targets=y, k=1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    X_test = mnist.test.images
    X_test = tf.reshape(X_test, shape=[-1, n_steps, n_inputs])
    y_test = mnist.test.labels

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        X_batch = 0
        y_batch = 0
        for epoch in range(n_epoch):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                X_batch = X_batch.reshape([-1, n_steps, n_inputs])
                sess.run(train_op, feed_dict={X:X_batch, y:y_batch})
            accuracy_train = accuracy.eval(feed_dict = {X:X_batch, y:y_batch})
            accuracy_test = accuracy.eval(feed_dict = {X:X_test, y:y_test})
            print('epoch:', epoch, '\ttrain accuracy:', accuracy_train, '\t test accuracy:', accuracy_test)

    return

def deep_RNN():
    n_layers = 3
    n_neurons = 100
    basic_cell = rnn.BasicRNNCell(num_units=n_neurons)
    multi_layer_cell = rnn.MultiRNNCell(cells=[basic_cell] * n_layers)
    outputs, states = tf.nn.dynamic_rnn(cell=multi_layer_cell)
    return

def RNN_with_dropout():
    X = 0
    keep_prob = 0.5
    cell = rnn.BasicRNNCell(num_units=100)
    cell_dropout = rnn.DropoutWrapper(cell=cell, input_keep_prob=keep_prob)
    multi_layer_cell = rnn.MultiRNNCell(cells=[cell_dropout] * 3)
    outputs, states = tf.nn.dynamic_rnn(cell=multi_layer_cell, inputs = X)

# deep_RNN()
# Simple_RNN()
# Static_RNN()
# Static_RNN2()
# dynamic_RNN()
# dynamic_RNN_with_Length()
# RNN_with_MNIST()
test()
# output_projection_wrapper()
# tricker_RNN()