import tensorflow as tf
import collections
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils


# # collections.namedtuple()
# sess = tf.Session()
#
# t = tf.constant([[1, 1, 1], [2, 2, 2]])
# paddings = tf.constant([[0, 1,], [1, 2]])
# # 'constant_values' is 0.
# # rank of 't' is 2.
# a = tf.pad(t, paddings, "REFLECT")
# print(sess.run(a))
# print(a.get_shape())

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    'bla bla'


def subsampling(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, ksize, strides, scope=None):
    if strides == 1:
        return slim.conv2d(inputs, num_outputs, ksize,
                           stride=1, padding='SAME', scope=scope)
    else:
        total_padding = ksize - 1
        padd_begin = total_padding // 2
        padd_end = total_padding - padd_begin
        inputs = tf.pad(inputs, [[0, 0], [padd_begin, padd_end],
                                 [padd_begin, padd_end]], [0, 0])
        return slim.conv2d(inputs, num_outputs, ksize,
                           strides, 'VALID', scope=scope)


def stack_block_dense(net, blocks, outputs_collections=None):
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net, depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
            net = utils.collect_named_outputs(outputs_collections, sc.name, net)
    return


def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):

    batch_norm_parameters = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope([slim.conv2d],
                        weight_regularizer=slim.l2_regularizer(weight_decay),
                        weight_initializer=slim.variance_scaling_initializer(),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_parameters):

        with slim.arg_scope([slim.batch_norm], **batch_norm_parameters):
            with slim.arg_scope([slim.max_pool2d],
                                padding='SAME') as arg_sc:
                return arg_sc

def bottleneck(inputs, depth, depth_bottleneck, stride,
               outputs_collection=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
        pre_activation = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preactivation')

        if depth == depth_in:
            shortcut = subsampling(inputs, stride, scope='shortcut')
        else:
            shortcut = slim.conv2d(inputs, depth, [1, 1], stride,
                                   normalizer_fn=None,
                                   activation_fn=None,
                                   scope='shortcut')

        residual = slim.conv2d(pre_activation,
                               depth_bottleneck, [1, 1],
                               stride=1, scope='conv1')
        residual = conv2d_same(residual,
                               depth_bottleneck, 3,
                               stride, scope='conv2')
        residual = slim.conv2d(residual,
                               depth, [1, 1],
                               stride=1, scope='conv3')

        output = shortcut + residual

        return utils.collect_named_outputs(outputs_collection,
                                           sc.name, output)

def resnet_v2(inputs,
              blocks,
              num_classes=None,
              global_pool=None,
              include_root_block=True,
              reuse=True,
              scope=None):
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_point'
        net = inputs
        if include_root_block:
            with slim.arg_scope([slim.conv2d], activation_fn = None, normalizer_fn=None):
                net = conv2d_same(net, 64, 7, strides=2, scope='conv1')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
        net = stack_block_dense(net, blocks)
        net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')

        if global_pool:
            net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)

        if num_classes is not None:
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                              normalizer_fn=None, scope='logits')
        end_points = utils.convert_collection_to_dict(end_points_collection)

        if num_classes is not None:
            end_points['predictions'] = slim.softmax(net, scope='prediction')
        return net, end_points

def resnet_v2_50(inputs,
                 num_classes=None,
                 global_pool=True,
                 reuse=None,
                 scope='resnet_v2_50'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]

    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)


