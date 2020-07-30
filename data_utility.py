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

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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
    print("======Loading data======")
    train_data, train_labels = merge_batches()
    test_data, test_labels = merge_batches(Train=False)

    train_labels = one_hot_encode(train_labels)
    test_labels = one_hot_encode(test_labels)

    train_data = train_data.reshape(50000, 3, 32, 32).transpose(0,2,3,1)
    test_data = test_data.reshape(10000, 3, 32, 32).transpose(0,2,3,1)

    print("Train data:",np.shape(train_data), np.shape(train_labels))
    print("Test data :",np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]

    print("======Prepare Finished======")

    return train_data, train_labels, test_data, test_labels

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

def data_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train[:,:,:,0] = (x_train[:,:,:,0] - np.mean(x_train[:,:,:,0])) / np.std(x_train[:,:,:,0])
    x_train[:,:,:,1] = (x_train[:,:,:,1] - np.mean(x_train[:,:,:,1])) / np.std(x_train[:,:,:,1])
    x_train[:,:,:,2] = (x_train[:,:,:,2] - np.mean(x_train[:,:,:,2])) / np.std(x_train[:,:,:,2])

    x_test[:,:,:,0] = (x_test[:,:,:,0] - np.mean(x_test[:,:,:,0])) / np.std(x_test[:,:,:,0])
    x_test[:,:,:,1] = (x_test[:,:,:,1] - np.mean(x_test[:,:,:,1])) / np.std(x_test[:,:,:,1])
    x_test[:,:,:,2] = (x_test[:,:,:,2] - np.mean(x_test[:,:,:,2])) / np.std(x_test[:,:,:,2])

    return x_train, x_test

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
