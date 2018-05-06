import tensorflow as tf
import tensorflow.contrib as contrib

slim = contrib.slim

trunc_normal = lambda stddev:tf.truncated_normal_initializer(0.0, stddev)

def print_activations(t):
    print('op name:', t.op.name, 'output size:', t.get_shape().as_list())
    return

def inception_v3_arg_scope(weight_decay,
                           stddev=0.1,
                           batch_norm_var_collection='moving_vars'):
    batch_norm_params = {
        'decay'            : 0.9997,
        'epsilon'          : 0.001,
        'update_collection': tf.GraphKeys.UPDATE_OPS,

        'cariable_collections' : {
            'beta'           : None,
            'gamma'          : None,
            'moving_bean'    : [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection]
        }
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer = slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d],
                            weight_initializer  = tf.truncated_normal_initializer(stddev=stddev),
                            activation_fn       = tf.nn.relu,
                            normalizer_fn       = slim.batch_norm,
                            normalizer_params   = batch_norm_params
                            ) as sc:
            return sc

def inception_v3_base(inputs, scope=None):
    end_points = {}
    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            strides=1,
                            padding='VALID'):

            net = slim.conv2d(inputs,
                              num_outputs=32,
                              kernel_size=[3, 3],
                              stride=2,
                              scope='conv2d_1a_3x3')

            net = slim.conv2d(net,
                              num_outputs=32,
                              kernel_size=[3, 3],
                              padding='SAME',
                              scope='conv2d_2a_3x3')

            net = slim.conv2d(net,
                              num_outputs=64,
                              kernel_size=[3, 3],
                              scope='conv2d_2b_3x3')

            max_pool = slim.max_pool2d(net,
                                       kernel_size=[3, 3],
                                       stride=2,
                                       scope='MaxPool_3a_3x3')

            net = slim.conv2d(max_pool,
                              num_outputs=80,
                              kernel_size=[1, 1],
                              scope='conv2d_3b_3x3')

            net = slim.conv2d(net,
                              num_outputs=192,
                              kernel_size=[3, 3],
                              scope='conv2d_4a_3x3')

            max_pool = slim.max_pool2d(net,
                                       kernel_size=[3, 3],
                                       stride=2,
                                       scope='MaxPool_5a_3x3')
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('branch_0'):
                    branch_0 = slim.conv2d(max_pool,
                                           num_outputs=64,
                                           kernel_size=[1, 1],
                                           scope='Conv2d_0a_1x1')
                with tf.variable_scope('branch_1'):
                    branch_1_1 = slim.conv2d(max_pool,
                                             num_outputs=48,
                                             kernel_size=[1, 1],
                                             scope='Conv2d_1a_1x1')
                    branch_1_2 = slim.conv2d(branch_1_1,
                                             num_outputs=64,
                                             kernel_size=[1, 1],
                                             scope='Conv2d_1b_1x1')
                with tf.variable_scope('branch_2'):
                    branch_2_1 = slim.conv2d(max_pool,
                                             num_outputs=64,
                                             kernel_size=[1, 1],
                                             scope='Conv2d_2a_1x1')
                    branch_2_2 = slim.conv2d(branch_2_1,
                                             num_outputs=96,
                                             kernel_size=[3, 3],
                                             scope='Conv2d_2b_3x3')
                    branch_2_3 = slim.conv2d(branch_2_2,
                                             num_outputs=96,
                                             kernel_size=[3, 3],
                                             scope='Conv2d_2c_3x3')
                with tf.variable_scope('branch_3'):
                    branch_3_1 = slim.avg_pool2d(max_pool,
                                                 kernel_size=[3, 3],
                                                 scope='AvgPool_3a_3x3')
                    branch_3_2 = slim.conv2d(branch_3_1,
                                             num_outputs=32,
                                             kernel_size=[3, 3],
                                             scope='AvgPool_3a_3x3')

                net = tf.concat([branch_0, branch_1_2, branch_2_3, branch_3_2], 3)

            with tf.variable_scope('Mixed_5c'):
                with tf.variable_scope('branch_0'):
                    branch_0 = slim.conv2d(max_pool,
                                           num_outputs=64,
                                           kernel_size=[1, 1],
                                           scope='Conv2d_0a_1x1')
                with tf.variable_scope('branch_1'):
                    branch_1_1 = slim.conv2d(max_pool,
                                             num_outputs=48,
                                             kernel_size=[1, 1],
                                             scope='Conv2d_1a_1x1')
                    branch_1_2 = slim.conv2d(branch_1_1,
                                             num_outputs=64,
                                             kernel_size=[1, 1],
                                             scope='Conv2d_1b_1x1')
                with tf.variable_scope('branch_2'):
                    branch_2_1 = slim.conv2d(max_pool,
                                             num_outputs=64,
                                             kernel_size=[1, 1],
                                             scope='Conv2d_2a_1x1')
                    branch_2_2 = slim.conv2d(branch_2_1,
                                             num_outputs=96,
                                             kernel_size=[3, 3],
                                             scope='Conv2d_2b_3x3')
                    branch_2_3 = slim.conv2d(branch_2_2,
                                             num_outputs=96,
                                             kernel_size=[3, 3],
                                             scope='Conv2d_2c_3x3')
                with tf.variable_scope('branch_3'):
                    branch_3_1 = slim.avg_pool2d(max_pool,
                                                 kernel_size=[3, 3],
                                                 scope='AvgPool_3a_3x3')
                    branch_3_2 = slim.conv2d(branch_3_1,
                                             num_outputs=64,
                                             kernel_size=[3, 3],
                                             scope='AvgPool_3a_3x3')
                    '''此处与前面不同之处'''

                net = tf.concat([branch_0, branch_1_2, branch_2_3, branch_3_2], 3)


