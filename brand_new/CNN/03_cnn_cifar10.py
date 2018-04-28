# More Advanced CNN Model: CIFAR-10
#---------------------------------------
#
# In this example, we will download the CIFAR-10 images
# and build a CNN model with dropout and regularization
#
# CIFAR is composed ot 50k train and 10k test
# images that are 32x32.

import os
import sys
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from six.moves import urllib
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Change Directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Start a graph session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# sess = tf.Session()

'''
2.Now we'll declare some of the model parameters. Our batch size will
be 128 (for train and test). We will output a status every 50
generations and run for a total of 20,000 generations. Every 500
generations, we'll evaluate on a batch of the test data. We'll then
declare some image parameters, height and width, and what size the
random cropped images will take. There are three channels (red,
green, and blue), and we have ten different targets. Then, we'll declare
where we will store the data and image batches from the queue:
'''
# Set model parameters
data_dir = 'D:\workspace\python\datasets\cifar10'
batch_size = 128
output_every = 50
generations = 20000
eval_every = 500
image_height = 32
image_width = 32
crop_height = 24
crop_width = 24
num_channels = 3
num_targets = 10
extract_folder = 'cifar-10-batches-bin'

'''
3.It is recommended to lower the learning rate as we progress towards a
good model, so we will exponentially decrease the learning rate: the
initial learning rate will be set at 0.1, and we will exponentially
decrease it by a factor of 10% every 250 generations.TensorFlow
does accept a staircase argument which only updates the learning
rate:
'''
learning_rate = 0.1
lr_decay = 0.1
num_gens_to_wait = 250.

'''
4. Now we'll set up parameters so that we can read in the binary CIFAR-
10 images:
'''
image_vec_length = image_height * image_width * num_channels
record_length = 1 + image_vec_length  # ( + 1 for "the 0-9 label")

'''
5. Next, we'll set up the data directory and the URL to download the
CIFAR-10 images, if we don't have them already:
'''

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
cifar10_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

data_file = os.path.join(data_dir, 'cifar-10-binary.tar.gz')
if os.path.isfile(data_file):
    pass
else:
    # Download file
    def progress(block_num, block_size, total_size):
        progress_info = [cifar10_url, float(block_num * block_size) / float(total_size) * 100.0]
        print('\r Downloading {} - {:.2f}%'.format(*progress_info), end="")
    filepath, _ = urllib.request.urlretrieve(cifar10_url, data_file, progress)
    # Extract file
    tarfile.open(filepath, 'r:gz').extractall(data_dir)
print("done")

'''
6. We'll set up the record reader and return a randomly distorted image
with the following read_cifar_files() function. First, we need to
declare a record reader object that will read in a fixed length of bytes.
After we read the image queue, we'll split apart the image and label.
Finally, we will randomly distort the image with TensorFlow's built in
image modification functions:
'''
def read_cifar_files(filename_queue, distort_iamges = True):
    reader = tf.FixedLengthRecordReader(record_bytes=record_length)
    key, record_string = reader.read(filename_queue)
    record_bytes = tf.decode_raw(record_string, tf.int8) # decode_raw操作可以将一个字符串转换为一个uint8的张量。
    image_label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int32)

    # Extract image
    image_extracted = tf.reshape(tf.slice(record_bytes, [1], [image_vec_length]),
                                 [num_channels, image_height, image_width])

    #reshape image
    image_uint8image = tf.transpose(image_extracted, [1, 2, 0])
    reshaped_iamge = tf.cast(image_uint8image, tf.float32)

    # Randomly Crop image
    final_image = tf.image.resize_image_with_crop_or_pad(reshaped_iamge, crop_width, crop_height)

    if distort_iamges:
        # Randomly flip the image horizontally, change the brightness and contrast
        final_image = tf.image.random_flip_left_right(final_image)
        final_image = tf.image.random_brightness(final_image, max_delta=63)
        final_image = tf.image.random_contrast(final_image, lower=0.2, upper=1.8)

    # Normalize whitening
    final_image = tf.image.per_image_standardization(final_image)
    return (final_image, image_label)

'''
7. Now we'll declare a function that will populate our image pipeline for
the batch processor to use. We first need to set up the file list of
images we want to read through, and to define how to read them with
an input producer object, created through prebuilt TensorFlow      
functions. The input producer can be passed into the reading function
that we created in the preceding step, read_cifar_files(). We'll then
set a batch reader on the queue, shuffle_batch():
'''
def input_pipeline(batch_size, train_logical=True):
    if train_logical:
        files = [os.path.join(data_dir, extract_folder, 'data_batch_{}.bin'.format(i)) for i in range(1, 6)]
    else:
        files = [os.path.join(data_dir, extract_folder, 'test_batch.bin')]
    filename_queue = tf.train.string_input_producer(files)
    image, label = read_cifar_files(filename_queue)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size, capacity, min_after_dequeue)
    return (example_batch, label_batch)
'''
Note
It is important to set the min_after_dequeue properly. This parameter
is responsible for setting the minimum size of an image buffer for
sampling. The official TensorFlow documentation recommends setting
it to (#threads + error margin)*batch_size. Note that setting it to a
larger size results in more uniform shuffling, as it is shuffling from a
larger set of data in the queue, but that more memory will also be used
in the process.
'''

'''
8. Next, declare our model function. 
The model has two convolutional layers, followed by three fully connected layers. 
To make variable declaration easier, we'll start by declaring two variable
functions. The two convolutional layers will create 64 features each.
The first fully connected layer will connect the 2nd convolutional
layer with 384 hidden nodes. The second fully connected operation
will connect those 384 hidden nodes to 192 hidden nodes. The final
hidden layer operation will then connect the 192 nodes to the 10
output classes we are trying to predict. See the following inline
comments marked with #:
'''

# for test
# files = [os.path.join(data_dir, extract_folder, 'data_batch_{}.bin'.format(i)) for i in range(1,6)]
# filename_queue = tf.train.string_input_producer(files)
# image, label = read_cifar_files(filename_queue)

def cifar_cnn_model(input_images, batch_size, train_logical=True):
    def truncated_normal_var(name, shape, dtype):
        return tf.get_variable(name=name, shape=shape, dtype=dtype)
    initializer = tf.truncated_normal_initializer(stddev=0.05)
    def zero_var(name, shape, dtype):
        return (tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=initializer))

    # First Convolutional Layer
    with tf.variable_scope('conv1') as scope:
        # Conv_kernel is 5x5 for all 3 colors and we will create 64 features

        # Initialize and add the bias term
        conv1_kernel = truncated_normal_var(name='conv_kernel1', shape=[5, 5, 3, 64], dtype=tf.float32)
        conv1_bias = zero_var(name='conv_bias1', shape=[64], dtype=tf.float32)

        # We convolve across the image with a stride size of 1
        conv1 = tf.nn.conv2d(input_images, conv1_kernel, [1, 1, 1, 1], padding='SAME')

        conv1_add_bias = tf.nn.bias_add(conv1, conv1_bias)

        # ReLU element wise
        relu_conv1 = tf.nn.relu(conv1_add_bias)

        # max pooling
        pool1 = tf.nn.max_pool(relu_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_layer1')

        # Local Response Normalization
        norm1 = tf.nn.lrn(pool1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm1')

    # Second Convolutional Layer
    with tf.variable_scope('conv2') as scope:
        # Conv_kernel is 5x5 for all 3 colors and we will create 64 features
        conv2_kernel = truncated_normal_var(name='conv_kernel2', shape=[5, 5, 64, 64], dtype=tf.float32)
        conv2_bias = zero_var(name='conv_bias2', shape=[64], dtype=tf.float32)

        # Convolve filter across prior output with stride size of 1
        conv2 = tf.nn.conv2d(norm1, conv2_kernel, [1, 1, 1, 1], padding='SAME')

        conv2_add_bias = tf.nn.bias_add(conv2, conv2_bias)

        # ReLU element wise
        relu_conv2 = tf.nn.relu(conv2_add_bias)

        # max pooling
        pool2 = tf.nn.max_pool(relu_conv2, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_layer2')

        # Local Response Normalization (parameters from paper)
        norm2 = tf.nn.lrn(pool2, depth_radius=5, bias=2., alpha=1e-3, beta=0.75, name='norm2')

        # Reshape output into a single matrix for multiplication for the fully connected layers
        reshaped_output = tf.reshape(norm2, [batch_size, -1])
        reshaped_dim = reshaped_output.get_shape()[1].value

    # First Fully Connected Layer
    with tf.variable_scope('full1') as scope:
        # Fully connected layer will have 384 outputs.
        full1_weight = truncated_normal_var(name='full_weight1', shape=[reshaped_dim, 384], dtype=tf.float32)
        full1_bias = zero_var(name='full_bias1', shape=[384], dtype=tf.float32)

        full_layer1 = tf.nn.relu(tf.add(tf.matmul(reshaped_output, full1_weight), full1_bias))

    # Second Fully Connected Layer
    with tf.variable_scope('full2') as scope:
        # Fully connected layer will have 192 outputs.
        full2_weight = truncated_normal_var(name='full_weight2', shape=[384, 192], dtype=tf.float32)
        full2_bias = zero_var(name='full_bias2', shape=[192], dtype=tf.float32)

        full_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1, full2_weight), full2_bias))
    # Final Fully Connected Layer -> 10 categories for output (num_targets)
    with tf.variable_scope('full3') as scope:
        # Fully connected layer will have 10 outputs.
        full3_weight = truncated_normal_var(name='full_weight3', shape=[192, 10], dtype=tf.float32)
        full3_bias = zero_var(name='full_bias3', shape=[10], dtype=tf.float32)

        full_layer3 = tf.nn.relu(tf.add(tf.matmul(full_layer2, full3_weight), full3_bias))

        final_output = full_layer3
        return final_output

'''
9. Now we'll create the loss function. We will use the softmax function
because a picture can only take on exactly one category, so the output
should be a probability distribution over the ten targets:
'''
def cifar_loss(logits, targets):
    # Get rid of extra dimensions and cast targets into integers
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # Calculate cross entropy from logits and targets
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    # Take the average loss across batch size
    loss = tf.reduce_mean(cross_entropy)
    return loss

'''
10. Next, we declare our training step. The learning rate will decrease in
an exponential step function:
'''
def train_step(loss_value, gen_num):
    # Our learning rate is an exponential decay (stepped down)
    model_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                     global_step=gen_num,
                                                     decay_steps=num_gens_to_wait,
                                                     decay_rate=lr_decay,
                                                     staircase=True)
    # Create optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate)

    # Initialize train step
    train_step = optimizer.minimize(loss_value)
    return train_step

'''
11. We must also have an accuracy function that calculates the accuracy
across a batch of images. We'll input the logits and target vectors, and
output an averaged accuracy. We can then use this for both the train
and test batches:
'''
def accuracy_of_batch(logits, targets):
    # Make sure targets are integers and drop extra dimensions
    targets = tf.squeeze(tf.cast(targets, tf.int32))

    # Get predicted values by finding which logit is the greatest
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)

    # Check if they are equal across the batch
    prediction_correctly = tf.equal(batch_predictions, targets)

    # Average the 1's and 0's (True's and False's) across the batch size
    accuracy = tf.reduce_mean(tf.cast(prediction_correctly, tf.float32))
    return accuracy

'''
12. Now that we have an imagepipeline function, we can initialize both
the training image pipeline and the test image pipeline:
'''
images, targets = input_pipeline(batch_size, train_logical=True)
test_images, test_targets = input_pipeline(batch_size, train_logical=False)

'''
13. Next, we'll initialize the model for the training output and the test
output. It is important to note that we must declare
scope.reuse_variables() after we create the training model so that,
when we declare the model for the test network, it will use the same
model parameters:
'''
with tf.variable_scope('model_definition') as scope:
    # Declare the training network model
    model_output = cifar_cnn_model(images, batch_size)

    # Use same variables within scope
    scope.reuse_variables()

    # Declare test model output
    test_output = cifar_cnn_model(test_images, batch_size)

'''
14. We can now initialize our loss and test accuracy functions. Then we'll
declare the generation variable. This variable needs to be declared as
non-trainable, and passed to our training function that uses it in the
learning rate exponential decay calculation:
'''
# learning rate exponential decay calculation:
print('Declare Loss Function.')
loss = cifar_loss(model_output, targets)

# Create accuracy function
accuracy = accuracy_of_batch(logits=test_output, targets=test_targets)
gen_num = tf.Variable(0, trainable=False)
train_op = train_step(loss, gen_num)

'''
15. We'll now initialize all of the model's variables and then start the image
pipeline by running the TensorFlow function, start_queue_runners().
When we start the train or test model output, the pipeline will feed
in a batch of images in place of a feed dictionary:
'''
# Initialize Variables
print('Initializing the Variables.')
init = tf.global_variables_initializer()
sess.run(init)

# Initialize queue (This queue will feed into the model, so no placeholders necessary)
tf.train.start_queue_runners(sess=sess)

'''
16. We now loop through our training generations and save the training
loss and the test accuracy:
'''
# Train CIFAR Model
print('Starting Training')

train_loss = []
test_accuracy = []
for i in range(generations):
    _, loss_value = sess.run([train_op, loss])

    if (i + 1) % output_every == 0:
        train_loss.append(loss_value)
        output = 'Generation {}: loss = {:.5f}'.format((i + 1), loss_value)
        print(output)

    if (i + 1) % eval_every == 0:
        [temp_acc] = sess.run([accuracy])
        test_accuracy.append(temp_acc)
        acc_output = ' --- Test Accuracy = {:.2f}%.'.format(100. * temp_acc)
        print(temp_acc)

'''
18. Finally, here is some matplotlib code that will plot the loss and test
accuracy over the course of the training:
'''
# Print loss and accuracy
# Matlotlib code to plot the loss and accuracies
eval_indices = range(0, generations, eval_every)
output_indices = range(0, generations, output_every)

# Plot loss over time
plt.plot(output_indices, train_loss, 'k-')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
plt.show()

# Plot accuracy over time
plt.plot(eval_indices, test_accuracy, 'k-')
plt.title('Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.show()