import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import tensorflow.examples.tutorials.mnist.input_data as input
# import tensorboard
import tensorflow.contrib as contrib

def iris():
    iris = load_iris()
    X = iris.data[:, (2,3)]
    print(np.shape(X))
    print(np.shape(iris.data))
    tf.nn.top_k()
    tf.nn.in_top_k()
    input.read_data_sets()

def train_MLP():
    mnist = input.read_data_sets("MNIST_data", one_hot=True)
    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels
    feature_columns = 784
    dnn_clf = tf.train
    print(np.shape(x_train))
    print(np.shape(y_train))
    print(np.shape(x_test))
    print(np.shape(y_test))


def train_DNN_with_MNIST():
    mnist = input.read_data_sets("MNIST_data", one_hot=True)
    #  定义各项参数
    n_input = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 100
    logit_num = 10
    root_logdir = "tflogs"
    logdir = "./tf-logs"
    # 输入数据，输出数据
    X = tf.placeholder(dtype=tf.float32,shape=(None, n_input), name="X")
    y = tf.placeholder(dtype=tf.int64,shape=(None, 10), name="y")

    # 构建DNN部分
    with tf.name_scope("DNN"):
        neuron1 = neuron_layer(X, n_hidden1, name="neuron1", activation="relu")
        neuron2 = neuron_layer(neuron1, n_hidden2, name="neuron2", activation="relu")
        logits = neuron_layer(neuron2, logit_num, name="logit")

    # 计算loss
    with tf.name_scope("loss"):
        x_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits, name="entropy")
        loss = tf.reduce_mean(x_entropy)

    # 训练优化部分
    learning_rate = 0.01
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="optimizer")
        train_op = optimizer.minimize(loss=loss)

    # 损失部分
    # correct = tf.nn.in_top_k(predictions=logits,targets=y, k=1, name="correct")
    with tf.name_scope("eval"):
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        accuracy_summary = tf.summary.scalar('acc', accuracy)

    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    # DNN运行部分
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        batch_size = 50
        sess.run(init)
        for epoch in range(50):
            for iteration in range(mnist.train.num_examples // batch_size):
                x_batches, y_batches = mnist.train.next_batch(batch_size=batch_size)
                sess.run(train_op, feed_dict={X: x_batches, y: y_batches})
                if iteration % 10 == 0:
                    summary_str = sess.run(accuracy_summary, feed_dict={X: x_batches, y: y_batches})
                    file_writer.add_summary(summary_str, epoch)
            acc_train = sess.run(accuracy, feed_dict={X: mnist.train.images, y: mnist.train.labels})
            acc_test = sess.run(accuracy, feed_dict={X: mnist.test.images, y: mnist.test.labels})
            print("epoch", epoch, " train acc:", acc_train, " test acc:", acc_test)
        save_path = saver.save(sess, "./model_result/my_model_final.ckpg")
        file_writer.close()
        return save_path

def neuron_layer(X_input, out_num, name, activation=None):
    in_num = int(X_input.get_shape()[1])
    stddev = 2 / np.sqrt(in_num)
    W = tf.Variable(tf.truncated_normal(stddev=stddev, shape=(in_num, out_num),name="weight"))
    b = tf.Variable(tf.zeros(shape=(out_num), name="bias"))
    z = tf.add(tf.matmul(X_input, W), b)
    if activation == "relu":
        return tf.nn.relu(z)
    else:
        return tf.nn.softmax(z)

# iris()
# train_MLP()
train_DNN_with_MNIST()