import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
class AGNAutoEncoder(object):
    def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(), scale = 0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.train_scale = scale

        network_weights = self._initial_weights()
        self.weights = network_weights
        self.x = tf.placeholder(dtype = tf.float32, shape=[None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input, )),
                                                     self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        self.cost = tf.reduce_mean(tf.square(self.reconstruction - self.x))
        self.cost = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))

        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def xaiver_init(self, fan_in, fan_out, constant = 1):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        value = tf.random_uniform(shape=(fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
        return value

    def _initial_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(self.xaiver_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros(shape=[self.n_hidden], dtype=tf.float32))

        all_weights['w2'] = tf.Variable(tf.zeros(shape=[self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros(shape=[self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        cost, _ = self.sess.run((self.cost, self.optimizer),
                                feed_dict={self.x:X, self.scale:self.train_scale})
        return cost

    def cost_test(self, X):
        cost = self.sess.run((self.cost),
                                feed_dict={self.x: X, self.scale: self.train_scale})
        return cost

    def transform(self, X):
        hidden_feature = self.sess.run((self.hidden),
                                feed_dict={self.x: X, self.scale: self.train_scale})
        return hidden_feature

    def generate(self, hidden = None):
        if hidden == None:
            hidden = np.random.normal(size=self.weights['b1'])
        reconstruction = self.sess.run(self.reconstruction, feed_dict={self.hidden:hidden})
        return reconstruction

    def reconstruction(self, X):
        reconstruction = self.sess.run(self.reconstruction,
                                       feed_dict={self.x: X, self.scale: self.train_scale})
        return reconstruction

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])

    def standard_scale(self, X_train, X_test):
        preprosessor = prep.StandardScaler().fit(X_train)
        X_train = preprosessor.transform(X_train)
        X_test = preprosessor.transform(X_test)
        return X_train, X_test

    def get_random_block_from_data(self, data, batch_size):
        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index : (start_index + batch_size)]

mnist_data = input_data.read_data_sets('CNN/temp', one_hot=True)
auto_encoder = AGNAutoEncoder(n_input=784,
                              n_hidden=200,
                              transfer_function=tf.nn.softplus,
                              optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                              scale=0.01)
X_train, X_test = auto_encoder.standard_scale(mnist_data.train.images, mnist_data.test.images)

num_samples = int(mnist_data.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

for epoch in range(training_epochs):
    avg_cost = 0
    for iteration in range(int(num_samples / batch_size)):
        batch_xs = auto_encoder.get_random_block_from_data(X_train, batch_size)

        cost = auto_encoder.partial_fit(batch_xs)

        avg_cost += cost / num_samples * batch_size

    if epoch % display_step == 0:
        print('epoch:', '%04d' %(epoch + 1), 'cost:', '{:.9f}'.format(avg_cost))

print('Total cost:' + str(auto_encoder.cost_test(X_test)))


