# -*- coding:utf-8 -*-

import os
import sys
import time
import pickle
import random
import math
import numpy as np
import tensorflow as tf

class_num       = 10
image_size      = 32
img_channels    = 3


# ========================================================== #
# ├─ prepare_data()
#  ├─ download training data if not exist by download_data()
#  ├─ load data by load_data()
#  └─ shuffe and return data
# ========================================================== #


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def merge_batches_100(Train=True):
    if Train:
        fileName = "train"
        data = unpickle(fileName)
        features = data[b'data']
        labels = np.array(data[b'fine_labels'])
    else:
        fileName = "test"
        data = unpickle(fileName)
        features = data[b'data']
        labels = np.array(data[b'fine_labels'])

    return features, labels

def merge_batches(Train=True):
    '''
    Description: Merge batches of CIFAR-10 data pickles
    Params: num_to_load = number of batches of CIFAR-10 to load and merge
    Outputs: merged features and labels from specified no. of batches of CIFAR-10
    '''
    if Train:
        for i in range(5):
            fileName = "data_batch_" + str(i + 1)
            data = unpickle(fileName)
            if i == 0:
                features = data[b'data']
                labels = np.array(data[b'labels'])
            else:
                features = np.append(features, data[b"data"], axis=0)
                labels = np.append(labels, data[b"labels"], axis=0)
    else:
        fileName = "test_batch"
        data = unpickle(fileName)
        features = data[b'data']
        labels = np.array(data[b'labels'])

    return features, labels

def one_hot_encode(data):
    one_hot = np.zeros((data.shape[0], class_num))
    one_hot[np.arange(data.shape[0]), data] = 1
    return one_hot

def prepare_data():
    #print("======Loading data======")
    train_data, train_labels = merge_batches()
    test_data, test_labels = merge_batches(Train=False)

    #train_data, train_labels = merge_batches_100()
    #test_data, test_labels = merge_batches_100(Train=False)

    train_labels = one_hot_encode(train_labels)
    test_labels = one_hot_encode(test_labels)

    train_data = train_data.reshape(50000, 3, 32, 32).transpose(0,2,3,1)
    test_data = test_data.reshape(10000, 3, 32, 32).transpose(0,2,3,1)

    #print("Train data:",np.shape(train_data), np.shape(train_labels))
    #print("Test data :",np.shape(test_data), np.shape(test_labels))
    #print("======Load finished======")

    #print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]

    #print("======Prepare Finished======")

    return train_data, train_labels, test_data, test_labels


# ========================================================== #
# ├─ _random_crop()
# ├─ _random_flip_leftright()
# ├─ data_augmentation()
# └─ color_preprocessing()
# batch
# layer
# ========================================================== #
def _random_crop(batch, crop_shape, padding=None):
        oshape = np.shape(batch[0])
        if padding:
            oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
        new_batch = []
        npad = ((padding, padding), (padding, padding), (0, 0))
        for i in range(len(batch)):
            new_batch.append(batch[i])
            if padding:
                new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                          mode='constant', constant_values=0)
            nh = random.randint(0, oshape[0] - crop_shape[0])
            nw = random.randint(0, oshape[1] - crop_shape[1])
            new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                                        nw:nw + crop_shape[1]]
        return new_batch

def _random_flip_leftright(batch):
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                batch[i] = np.fliplr(batch[i])
        return batch

def data_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train[:,:,:,0] = (x_train[:,:,:,0] - np.mean(x_train[:,:,:,0])) / np.std(x_train[:,:,:,0])
    x_train[:,:,:,1] = (x_train[:,:,:,1] - np.mean(x_train[:,:,:,1])) / np.std(x_train[:,:,:,1])
    x_train[:,:,:,2] = (x_train[:,:,:,2] - np.mean(x_train[:,:,:,2])) / np.std(x_train[:,:,:,2])

    x_test[:,:,:,0] = (x_test[:,:,:,0] - np.mean(x_test[:,:,:,0])) / np.std(x_test[:,:,:,0])
    x_test[:,:,:,1] = (x_test[:,:,:,1] - np.mean(x_test[:,:,:,1])) / np.std(x_test[:,:,:,1])
    x_test[:,:,:,2] = (x_test[:,:,:,2] - np.mean(x_test[:,:,:,2])) / np.std(x_test[:,:,:,2])

    return x_train, x_test
'''

def data_preprocessing(x_train):
    x_train = x_train.astype('float32')

    x_train[:,:,:,0] = (x_train[:,:,:,0] - np.mean(x_train[:,:,:,0])) / np.std(x_train[:,:,:,0])
    x_train[:,:,:,1] = (x_train[:,:,:,1] - np.mean(x_train[:,:,:,1])) / np.std(x_train[:,:,:,1])
    x_train[:,:,:,2] = (x_train[:,:,:,2] - np.mean(x_train[:,:,:,2])) / np.std(x_train[:,:,:,2])

    return x_train
'''

def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32,32], 4)
    return batch

def next_batch(batch_size, train, label, number):
    i = random.randrange(0, number)
    if i < number-batch_size:
        batch_data = train[i:i+batch_size,:,:,:]
        batch_label = label[i:i+batch_size,:]
    else:
        batch_data = np.concatenate( (train[i:number,:,:,:] , train[0:batch_size-(number-i),:,:,:] ) )
        batch_label = np.concatenate( (label[i:number,:] , label[0:batch_size-(number-i),:] ) )
    return batch_data, batch_label
'''
def train_next_batch(batch_size):
    i = random.randrange(0, train_number)
    if i < train_number-batch_size:
        batch_data = train_x[i:i+batch_size,:,:,:]
        batch_label = train_y[i:i+batch_size,:]
    else:
        batch_data = np.concatenate( (train_x[i:train_number,:,:,:] , train_x[0:batch_size-(train_number-i),:,:,:] ) )
        batch_label = np.concatenate( (train_y[i:train_number,:] , train_y[0:batch_size-(train_number-i),:] ) )
    return batch_data, batch_label
def test_next_batch(batch_size):
    i = random.randrange(0, test_number)
    if i < test_number-batch_size:
        batch_data = test_x[i:i+batch_size,:,:,:]
        batch_label = test_y[i:i+batch_size,:]
    else:
        batch_data = np.concatenate( (test_x[i:test_number,:,:,:] , test_x[0:batch_size-(test_number-i),:,:,:] ) )
        batch_label = np.concatenate( (test_y[i:test_number,:] , test_y[0:batch_size-(test_number-i),:] ) )
    return batch_data, batch_label
'''

def maxpool(x_tensor, pool_ksize, pool_strides):
    return tf.nn.max_pool(x_tensor, ksize = [1, pool_ksize[0], pool_ksize[1], 1], strides = [1, pool_strides[0], pool_strides[1], 1], padding = 'VALID')

def avgpool(x_tensor, pool_ksize, pool_strides):
    return tf.nn.avg_pool(x_tensor, ksize = [1, pool_ksize[0], pool_ksize[1], 1], strides = [1, pool_strides[0], pool_strides[1], 1], padding = 'VALID')

def conv2d(x_tensor, conv_num_outputs, conv_ksize, conv_strides, weight):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # TODO: Implement Function
    x_shape = x_tensor.get_shape().as_list()
    n = conv_ksize[0] * conv_ksize[1] * conv_num_outputs
    weights = tf.get_variable(weight, shape=[conv_ksize[0], conv_ksize[1], x_shape[3], conv_num_outputs],
                              initializer=tf.variance_scaling_initializer())
    bias = tf.Variable(tf.zeros([conv_num_outputs]))

    L = tf.nn.conv2d(x_tensor, weights, strides = [1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    return L

def relu(L):
    return tf.nn.relu(L)

def fixed_padding(inputs, kernel_size):

  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides):

  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer())

def flatten(layer):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    return tf.contrib.layers.flatten(layer)

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    x_shape = x_tensor.get_shape().as_list()

    weights = tf.get_variable('weight', shape= [x_shape[1], num_outputs],
                              initializer=tf.variance_scaling_initializer())
    bias = tf.Variable(tf.zeros([num_outputs]))
    out = tf.add(tf.matmul(x_tensor, weights), bias)
    return out

# Functions for constructing layers
def dense_layer(inputs, name):
    with tf.variable_scope(name):
        f = flatten(inputs)
        out = output(f, class_num)
    return out

def dense_first_layer(inputs, num, name):
    with tf.variable_scope(name):
        f = flatten(inputs)
        out = output(f, num)
    return out

def dense_remain_layer(inputs, num, name):
    with tf.variable_scope(name):
        out = output(inputs, num)
    return out


def conv_layer(inputs, n_filter, kernel_size , stride_size,training,name):
    with tf.variable_scope(name):
        conv1 = conv2d_fixed_padding(inputs, n_filter , kernel_size, stride_size)
        conv1 = tf.layers.batch_normalization(conv1, training=training, name='batch_normalization')
        conv = relu(conv1)
    return conv

def conv_layer_sg(inputs, n_filter, kernel_size , stride_size,name):
    with tf.variable_scope(name):
        conv1 = conv2d_fixed_padding(inputs, n_filter , kernel_size, stride_size)
        conv = relu(conv1)
    return conv
