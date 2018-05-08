import tensorflow as tf
import collections
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils

batch_norm_parameters = {
    'is_training': True,
    'decay': 0.997,
    'epsilon': 1e-5,
    'scale': True,
    'updates_collections': tf.GraphKeys.UPDATE_OPS,
}

weight_decay = 0.0001
batch_norm_decay=0.997
w_regularizer = slim.l2_regularizer(weight_decay)
w_initializer = slim.variance_scaling_initializer()
act_fn = tf.nn.relu
norm_fn = slim.batch_norm
norm_params = batch_norm_parameters

def conv2d(inputs, num_outputs,
           ksize, strides,
           padding='SAME'):

    return slim.conv2d(inputs,
                       num_outputs=num_outputs,
                       kernel_size=ksize,
                       stride=strides,
                       padding=padding,
                       weights_regularizer=w_regularizer,
                       weights_initializer=w_initializer,
                       activation_fn= act_fn,
                       normalizer_fn=norm_fn,
                       normalizer_params=norm_params,)

def max_pool_2d(inputs, ksize, strides=None):
    return slim.max_pool2d(inputs=inputs,
                           kernel_size=ksize,
                           stride=strides,)

def batch_norm(inputs, activation_fn=None):
    return slim.batch_norm(inputs=inputs,
                           activation_fn=activation_fn,
                           is_training=True,
                           decay=batch_norm_decay,
                           epsilon=1e-5,
                           scale=True,
                           updates_collections=tf.GraphKeys.UPDATE_OPS)

def subsampling(inputs, strides):
    if strides == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], strides)


def conv2d_same(inputs, num_outputs, ksize, strides):

    if strides == 1:
        return conv2d(inputs, num_outputs, ksize, strides=1,
                           padding='SAME')
    else:
        total_padding = ksize - 1
        padd_begin = total_padding // 2
        padd_end = total_padding - padd_begin

        inputs = tf.pad(inputs, [[0, 0], [padd_begin, padd_end],
                                 [padd_begin, padd_end], [0, 0]])

        return conv2d(inputs, num_outputs, ksize,strides=strides,
                      padding='VALID')

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    'bla bla'

def bottleneck(inputs, strides, depth_bottleneck, depth):
    depth_in = utils.last_dimension(inputs.get_shape(),min_rank=4)
    preact = batch_norm(inputs, activation_fn=tf.nn.relu)

    if depth == depth_in:
        print('depth == depth_in')
        shortcut = subsampling(inputs, strides)
    else:
        print('depth != depth_in')
        shortcut = slim.conv2d(preact, depth,
                               [1, 1], strides,
                               normalizer_fn=None,
                               activation_fn=None,)
    print(shortcut.get_shape())
    residual = conv2d(preact, depth_bottleneck, 1, strides=1)

    residual = conv2d_same(residual, depth_bottleneck, 3, strides)

    residual = conv2d(residual, depth, 1, strides=1)
    print(residual.get_shape())

    output = residual + shortcut

    return output

def stack_dense_blocks(inputs, blocks):
    net = inputs
    for block in blocks:
        for i, unit in enumerate(block.args):
            depth, depth_bottleneck, strides = unit
            net = block.unit_fn(net,strides,depth_bottleneck,depth)
    return net

def resnet_v2(inputs,
              blocks,
              num_classes=None,
              include_root_block=True,
              global_pool=True,
              reuse=True,):

    net = inputs
    if include_root_block == True:
        net = conv2d_same(net, 64, 7, strides=2) #, scope=''
        net = max_pool_2d(net, ksize=[3, 3], strides=2) #, scope=''

    net = stack_dense_blocks(net, blocks)

    net = batch_norm(net, activation_fn=tf.nn.relu)

    if global_pool:
        net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)

    if num_classes is not None:
        net = slim.conv2d(net, num_classes, 1, 1,
                          activation_fn=None,
                          normalizer_fn=None)

        net = slim.softmax(net)

    return net

def resnet_v2_50(inputs,
                 num_classes=None,
                 global_pool=True,
                 reuse=None,):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]

    return resnet_v2(inputs, blocks, num_classes,
                     include_root_block=True,
                     global_pool=global_pool,
                     reuse=reuse)
batch_size = 1
height, width = 224, 224
inputs = tf.random_uniform((batch_size, height, width, 3))
net = resnet_v2_50(inputs, 1000)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sess.run(net)

