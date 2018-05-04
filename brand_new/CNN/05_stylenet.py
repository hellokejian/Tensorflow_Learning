# Using TensorFlow for Stylenet/NeuralStyle
#---------------------------------------
#
# We use two images, an original image and a style image
# and try to make the original image in the style of the style image.
#
# Reference paper:
# https://arxiv.org/abs/1508.06576
#
# Need to download the model 'imagenet-vgg-verydee-19.mat' from:
#   http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat

import scipy.io
import scipy.misc
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a graph session
sess = tf.Session()

'''
3. Then we can start a graph session and declare the locations of our two
images: the original image and the style image. For our purposes, we
will use the cover image of this book for the original image; for the
style image, we will use Starry Night by Vincent van Gough. Feel free
to use any two pictures you want here. If you choose to use these
pictures, they are available on the book's github site,
https://github.com/nfmcclure/tensorflow_cookbook (Navigate
tostyelnet section):
'''
# Image Files
original_image_file = 'pic/origin.jpg'
style_image_file = 'pic/starry_night.jpg'

'''
4. We'll set some parameters for our model: the location of the mat file,
weights, the learning rate, number of generations, and how frequently
we should output the intermediate image. For the weights, it helps to
highly weight the style image over the original image. These
hyperparameters should be tuned for changes in the desired result:
'''
# Saved VGG Network path under the current project dir.
vgg_path ='D:/workspace/python/datasets/vgg-net/imagenet-vgg-verydeep-19.mat'

# Default Arguments
original_image_weight = 5.0
style_image_weight = 500.0
regularization_weight = 100
learning_rate = 0.001
generations = 5000
output_generations = 250
beta1 = 0.9
beta2 = 0.999
'''
5. Now we'll load the two images with scipy and change the style image
to fit the original image dimensions:
'''
# Read in images
original_image = scipy.misc.imread(original_image_file)
style_image = scipy.misc.imread(style_image_file)

# Get shape of target and make the style image the same
target_shape = original_image.shape
style_image = scipy.misc.imresize(style_image, target_shape[1] / style_image.shape[1])

'''
6. From the paper, we can define the layers in order of how they
appeared. We'll use the author's naming convention:
'''
# VGG-19 Layer Setup
# From paper
vgg_layers = ['conv1_1', 'relu1_1',
              'conv1_2', 'relu1_2', 'pool1',
              'conv2_1', 'relu2_1',
              'conv2_2', 'relu2_2', 'pool2',
              'conv3_1', 'relu3_1',
              'conv3_2', 'relu3_2',
              'conv3_3', 'relu3_3',
              'conv3_4', 'relu3_4', 'pool3',
              'conv4_1', 'relu4_1',
              'conv4_2', 'relu4_2',
              'conv4_3', 'relu4_3',
              'conv4_4', 'relu4_4', 'pool4',
              'conv5_1', 'relu5_1',
              'conv5_2', 'relu5_2',
              'conv5_3', 'relu5_3',
              'conv5_4', 'relu5_4']

'''
7. Now we'll define a function that will extract the parameters from the
mat file:
'''
def extract_net_info(path_to_params):
    vgg_data = scipy.io.loadmat(path_to_params)
    normalization_matrix = vgg_data['normalization'][0][0][0]
    mat_mean = np.mean(normalization_matrix, axis=(0, 1))
    network_weights = vgg_data['layers'][0]
    return (mat_mean, network_weights)

'''8. From the loaded weights and the layer definitions, we can recreate
the network in TensorFlow with the following function. We'll loop
through each layer and assign the corresponding function with
appropriate weights and biases, where applicable:
'''
# Create the VGG-19 Network
def vgg_network(network_weights, init_image):
    network = {}
    image = init_image
    for i, layer in enumerate(vgg_layers):
        if layer[1] == 'c':
            weights, bias = network_weights[i][0][0][0][0]
            weights = tf.transpose(weights, [1, 0, 2, 3])
            bias = bias.reshape(-1)
            conv_layer = tf.nn.conv2d(image, tf.constant(weights), (1, 1, 1, 1), 'SAME')
            image = tf.nn.bias_add(conv_layer, bias)
        elif layer[1] == 'r':
            image = tf.nn.relu(image)
        else:
            image = tf.nn.max_pool(image, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME')
        network[layer] = image
    return (network)

# for i, layer in enumerate(vgg_layers):
#     print(i, '-', layer)

'''
9. The paper recommends a few strategies of assigning intermediate
layers to the original and style images. While we should keep relu4_2
for the original image, we can try different combinations of the other
reluX_1 layer outputs for the style image:
'''
# Here we define which layers apply to the original or style image
original_layer = 'relu4_2'
style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

'''
10. Next, we'll run the above function to get the weights and mean. We'll
also change the image shapes to have four dimensions by adding a
dimension of size one to the beginning. TensorFlow's image operations
act on four dimensions, so we must add the batch-size dimension:
'''
# Get network parameters
# normalization_mean, network_weights = extract_net_info(vgg_path)
#
# shape = (1,) + original_image.shape
# style_shape = (1,) + style_image.shape
# original_features = {}
# style_features = {}

vgg_data = scipy.io.loadmat(vgg_path)
# print(vgg_data['normalization'])
print(vgg_data['layers'])