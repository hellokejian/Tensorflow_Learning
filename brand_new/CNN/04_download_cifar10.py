# Download/Saving CIFAR-10 images in Inception format
#---------------------------------------
#
# In this script, we download the CIFAR-10 images and
# transform/save them in the Inception Retrianing Format
#
# The end purpose of the files is for retrianing the
# Google Inception tensorflow model to work on the CIFAR-10.

import os
import tarfile
import _pickle as cPickle
import numpy as np
import urllib.request
import scipy.misc

cifar_link = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
# data_dir = 'D:/workspace/python/datasets/cifar10'
data_dir = 'D:\workspace\python\datasets\cifar10'
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)
objects = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

'''
3. Now we'll download the CIFAR-10 .tar data file, and un-tar the file:
'''
# Download tar file
target_file = os.path.join(data_dir, 'cifar-10-python.tar.gz')
if not os.path.isfile(target_file):
    print('CIFAR-10 file not found. Downloading CIFAR data (Size = 163MB)')
    print('This may take a few minutes, please wait.')
    filename, headers = urllib.request.urlretrieve(cifar_link, target_file)

# Extract into memory
tar = tarfile.open(target_file)
tar.extractall(path=data_dir)
tar.close()

'''
4. We now create the necessary folder structure for training. The
temporary directory will have two folders, train_dir and
validation_dir. In each of these folders, we will create the ten subfolders
for each category:
'''
# Create train image folders
train_folder = 'train_dir'
if not os.path.isdir(os.path.join(data_dir, train_folder)):
    for i in range(10):
        folder = os.path.join(data_dir, train_folder, objects[i])
        os.makedirs(folder)
# Create test image folders
test_folder = 'validation_dir'
if not os.path.isdir(os.path.join(data_dir, test_folder)):
    for i in range(10):
        folder = os.path.join(data_dir, test_folder, objects[i])
        os.makedirs(folder)

# Extract images accordingly
data_location = os.path.join(data_dir, 'cifar-10-batches-py')
train_names = ['data_batch_' + str(x) for x in range(1, 6)]
test_names = ['test_batch']

def load_batch_from_file(file):
    file_conn = open(file, 'rb')
    image_dictionary = cPickle.load(file_conn, encoding='latin1')
    file_conn.close()
    return (image_dictionary)

'''
6. With the above dictionary, we will save each of the files in the correct
location with the following function:
'''
def save_iamges_from_dict(image_dict, folder='data_dir'):
    # image_dict.keys() = 'labels', 'filenames', 'data', 'batch_label'
    for ix, label in enumerate(image_dict['labels']):
        folder_path = os.path.join(data_dir, folder, objects[label])
        filename = image_dict['filenames'][ix]
        # Transform image data
        image_array = image_dict['data'][ix]
        image_array.resize([3, 32, 32])
        # Save image
        output_location = os.path.join(folder_path, filename)
        scipy.misc.imsave(output_location, image_array.transpose())
    return

'''
7. With the preceding functions, we can loop through the downloaded
data files and save each image to the correct location:
'''
for file in train_names:
    print('Saving images from file: {}'.format(file))
    file_location = os.path.join(data_dir, 'cifar-10-batches-py', file)
    image_dict = load_batch_from_file(file_location)
    save_iamges_from_dict(image_dict, train_folder)


# Sort test images
for file in test_names:
    print('Saving images from file: {}'.format(file))
    file_location = os.path.join(data_dir, 'cifar-10-batches-py', file)
    image_dict = load_batch_from_file(file_location)
    save_iamges_from_dict(image_dict, train_folder)

# Create labels file
cifar_labels_file = os.path.join(data_dir, 'cifar10_labels.txt')
print('Writing labels file, {}'.format(cifar_labels_file))
with open(cifar_labels_file, 'w') as labels_file:
    for item in objects:
        labels_file.write("{}\n".format(item))

