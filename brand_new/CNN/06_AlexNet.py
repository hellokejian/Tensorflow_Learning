import tensorflow as tf

batch_size = 32
num_batches = 100

image_height = 224
image_width = 224
num_channels = 3
images_train = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, num_channels])
labels_train = tf.placeholder(dtype=tf.int64, shape=[None])

def print_activations(t):
    print('op name:', t.op.name, 'output size:', t.get_shape().as_list())
    return

def weight_initializer(shape, dtype, mean=0., stddev=1.):
    return tf.Variable(tf.truncated_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype))

def inference(images):

    parameters = []
    with tf.name_scope('conv1') as scope:
        conv1_kernel = weight_initializer([11, 11, 3, 64], tf.float32, 0, 0.1e-1)

        conv1 = tf.nn.conv2d(images, conv1_kernel, strides=[1, 4, 4, 1], padding='SAME')

        conv1_bias = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[64]), trainable=True, name='bias')

        conv1_add_bias = tf.nn.bias_add(conv1, conv1_bias)

        conv1_relu = tf.nn.relu(conv1_add_bias)

        print_activations(conv1_relu)

        parameters += [conv1_kernel, conv1_bias]

        conv1_norm = tf.nn.lrn(conv1_relu, depth_radius=4, bias=1, alpha=0.001 / 9, beta=0.75)

        conv1_maxpool = tf.nn.max_pool(conv1_norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        print_activations(conv1_maxpool)

    with tf.name_scope('conv2') as scope:
        conv2_kernel = weight_initializer([5, 5, 64, 192], tf.float32, 0, 0.1e-1)

        conv2 = tf.nn.conv2d(conv1_maxpool, conv2_kernel, strides=[1, 1, 1, 1], padding='SAME')

        conv2_bias = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[192]), trainable=True, name='bias')

        conv2_add_bias = tf.nn.bias_add(conv2, conv2_bias)

        conv2_relu = tf.nn.relu(conv2_add_bias)

        print_activations(conv2_relu)

        parameters += [conv2_kernel, conv2_bias]

        conv2_norm = tf.nn.lrn(conv2_relu, depth_radius=4, bias=1, alpha=0.001 / 9, beta=0.75)

        conv2_maxpool = tf.nn.max_pool(conv2_norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        print_activations(conv2_maxpool)

    with tf.name_scope('conv3') as scope:
        conv3_kernel = weight_initializer([3, 3, 192, 384], tf.float32, 0, 0.1e-1)

        conv3 = tf.nn.conv2d(conv2_maxpool, conv3_kernel, strides=[1, 1, 1, 1], padding='SAME')

        conv3_bias = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[384]), trainable=True, name='bias')

        conv3_add_bias = tf.nn.bias_add(conv3, conv3_bias)

        conv3_relu = tf.nn.relu(conv3_add_bias)

        print_activations(conv3_relu)

        parameters += [conv3_kernel, conv3_bias]

        # conv3_norm = tf.nn.lrn(conv3_relu, depth_radius=4, bias=1, alpha=0.001 / 9, beta=0.75)
        #
        # conv3_maxpool = tf.nn.max_pool(conv3_norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        #
        # print_activations(conv3_maxpool)
    with tf.name_scope('conv4') as scope:
        conv4_kernel = weight_initializer([3, 3, 384, 256], tf.float32, 0, 0.1e-1)

        conv4 = tf.nn.conv2d(conv3_relu, conv4_kernel, strides=[1, 1, 1, 1], padding='SAME')

        conv4_bias = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[256]), trainable=True, name='bias')

        conv4_add_bias = tf.nn.bias_add(conv4, conv4_bias)

        conv4_relu = tf.nn.relu(conv4_add_bias)

        print_activations(conv4_relu)

        parameters += [conv4_kernel, conv4_bias]

        # conv4_norm = tf.nn.lrn(conv4_relu, depth_radius=4, bias=1, alpha=0.001 / 9, beta=0.75)
        #
        # conv4_maxpool = tf.nn.max_pool(conv4_norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        #
        # print_activations(conv4_maxpool)

    with tf.name_scope('conv5') as scope:
        conv5_kernel = weight_initializer([3, 3, 256, 256], tf.float32, 0, 0.1e-1)

        conv5 = tf.nn.conv2d(conv4_relu, conv5_kernel, strides=[1, 1, 1, 1], padding='SAME')

        conv5_bias = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[256]), trainable=True, name='bias')

        conv5_add_bias = tf.nn.bias_add(conv5, conv5_bias)

        conv5_relu = tf.nn.relu(conv5_add_bias)

        print_activations(conv5_relu)

        parameters += [conv5_kernel, conv5_bias]

        # conv5_norm = tf.nn.lrn(conv5_relu, depth_radius=4, bias=1, alpha=0.001 / 9, beta=0.75)

        conv5_maxpool = tf.nn.max_pool(conv5_relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        print_activations(conv5_maxpool)
    return



inference(images_train)
