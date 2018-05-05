import tensorflow as tf
import tensorflow.contrib.layers as layer
def conv_op(input_op, name, kh, kw, n_features, dh, dw, p):
    num_channels = input_op.get_shape()[-1].value
    with tf.name_scope(name)as scope:
        kernels = tf.get_variable(name=scope + 'kernels', dtype=tf.float32,
                                  shape=[kh, kw, num_channels, n_features],
                                  initializer=layer.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernels, [1, dh, dw, 1],
                            padding='SAME')
        conv_bias = tf.Variable(tf.constant(0., tf.float32, [n_features], name='bias'))

        conv_add_bias = tf.nn.bias_add(conv, conv_bias)

        activation = tf.nn.relu(conv_add_bias)

        p += [kernels, conv_bias]

        return activation

def fc_op(input_op, name, n_output, p, activation=None):
    batch_size = input_op.get_shape()[0].value
    input_op_flatten = tf.reshape(input_op, [batch_size, -1], name='input_flatten')
    n_input = input_op_flatten.get_shape()[1].value
    with tf.name_scope(name)as scope:
        fc_weights = tf.get_variable(scope + 'weights', [n_input, n_output],
                                     tf.float32,initializer=layer.xavier_initializer())
        fc_bias = tf.Variable(tf.constant(0., tf.float32, [n_output]), name='fc_bias')

        activation = tf.nn.relu_layer(input_op_flatten, fc_weights, fc_bias, name=scope)

        p += [fc_weights, fc_bias]

        return  activation

def max_pool_op(conv_op, name, kh, kw, sh, sw, padding='SAME'):
    return tf.nn.max_pool(conv_op,
                          name=name,
                          ksize=(1, kh, kw, 1),
                          strides=(1, sh, sw, 1),
                          padding=padding)

def inference(images):

    return

