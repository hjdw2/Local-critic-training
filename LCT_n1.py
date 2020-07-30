import tensorflow as tf
import numpy as np
import random
import os
import sys
import time
from tqdm import tqdm
from data_utility import *
from resnet_model import *
sys.path.append(os.pardir)

cwd = os.getcwd()
cwd = cwd.replace("\\", "/")
tf.set_random_seed(1)

class_num = 10 # CIFAR10
train_number = 50000
test_number = 10000
weight_decay = 0.0005
iterations = 80000
batch_size = 128
validation_checkpoint = 400 # 1 epoch
block_fn = bottleneck_block
layers = [3, 4, 23, 3]

#Input data
train_x, train_y, test_x, test_y = prepare_data()
train_x, test_x = data_preprocessing(train_x, test_x)

X = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name="data")
Y = tf.placeholder(tf.float32, shape=(None, class_num), name="labels") # Target
learning_rate = tf.placeholder(tf.float32, shape =[])
is_training = tf.placeholder(tf.bool, shape =[])

with tf.device("/gpu:0"):
    with tf.variable_scope('res_block_1'):
        h2 = conv2d_fixed_padding(inputs=X, filters=64, kernel_size=3, strides=1)
        h2 = block_layer(inputs=h2, filters=64, block_fn=block_fn, blocks=layers[0], strides=1, is_training=is_training, name='block_layer1')
        h2 = block_layer(inputs=h2, filters=128, block_fn=block_fn, blocks=layers[1], strides=2, is_training=is_training, name='block_layer2')
        h2_1 = block_layer(inputs=h2, filters=256, block_fn=block_fn, blocks=10, strides=2, is_training=is_training, name='block_layer3')

    with tf.variable_scope('LC_block_1'):
        h_LC1 = LC_block(inputs=h2_1, name='LC_block_1')
        h_LC1 = dense_layer(h_LC1, name="LC1_2")

with tf.device("/gpu:1"):
    with tf.variable_scope('res_block_2'):
        h3 = block_layer_remain(inputs=h2_1, filters=256, block_fn=block_fn, blocks=13, strides=1, is_training=is_training, name='block_layer4')
        h3 = block_layer(inputs=h3, filters=512, block_fn=block_fn, blocks=layers[3], strides=2, is_training=is_training, name='block_layer5')
        h3 = batch_norm_relu(h3, is_training)
        logits = dense_layer(inputs=h3, name='logits')

# Collections of trainable variables in each block
layer_vars = [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="res_block_1"),
tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="res_block_2")]
LC_vars = [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="LC_block_1")]

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# Trainer for each block
def Block_Trainer(n, h_n, loss_layer=None):
    if n == 2:
        layer_grads = tf.gradients(h_n + tf.add_n([tf.nn.l2_loss(var) for var in layer_vars[n-1]]) * weight_decay ,layer_vars[n-1])
    else:
        layer_grads = tf.gradients(h_n + tf.add_n([tf.nn.l2_loss(var) for var in layer_vars[n-1]]) * weight_decay ,layer_vars[n-1], tf.gradients(loss_layer, h_n))
    layer_gv = list(zip(layer_grads,layer_vars[n-1]))
    layer_opt = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum = 0.9,use_nesterov=True).apply_gradients(layer_gv)
    return layer_opt

with tf.device("/gpu:1"):
    pred_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
    print_pred_loss = tf.reduce_mean(pred_loss)
    Block2_opt = Block_Trainer(2, tf.reduce_mean(pred_loss))
    correct_preds = tf.equal(tf.argmax(logits,1), tf.argmax(Y,1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds,tf.int32))

with tf.device("/gpu:0"):
    pred_loss_LC1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=h_LC1, labels=Y)
    LC1_opt = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(tf.reduce_mean(tf.abs(pred_loss - pred_loss_LC1)), var_list=LC_vars[0])
    Block1_opt = Block_Trainer(1, h2_1, loss_layer=tf.reduce_mean(pred_loss_LC1))
    correct_preds_1 = tf.equal(tf.argmax(h_LC1,1), tf.argmax(Y,1))
    accuracy_1 = tf.reduce_sum(tf.cast(correct_preds_1,tf.int32))

config = tf.ConfigProto()
LC_sess = tf.Session(config=config)
with LC_sess.as_default():
    LC_sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver.restore(LC_sess, cwd+"/model/checkpoint.ckpt-1")
    best_acc = 0
    for i in tqdm(range(1,iterations+1)):
        lr = None
        if i < 40000:
            lr = 0.1
        elif i >=40000 and i < 60000:
            lr = 0.01
        elif i >=60000 and i < 80000:
            lr = 0.001
        elif i>= 80000:
            lr=  0.0001

        data, target = next_batch(batch_size, train_x, train_y, train_number)
        data = data_augmentation(data)
        LC_sess.run([Block2_opt, Block1_opt, LC1_opt, extra_update_ops], feed_dict={X:data,Y:target, learning_rate:lr, is_training:True})

        if i % validation_checkpoint == 0:
            acc_ = LC_sess.run(accuracy, feed_dict={X:data,Y:target, is_training:False})
            print('Epoch: %d, Train sample Accuracy: %.4f' % (i/validation_checkpoint, acc_/batch_size)) # 50000 - 128*390 = 80

            test_total_loss = 0
            test_total_acc = 0
            for data, target in batch_features_labels(test_x, test_y, batch_size=200):
                test_loss, test_acc = LC_sess.run([print_pred_loss, accuracy], feed_dict={X:data,Y:target, learning_rate:lr, is_training:False})

                test_total_loss += test_loss
                test_total_acc += test_acc/200

            if test_total_acc > best_acc:
                print('Saving..')
                save_path = saver.save(LC_sess,cwd+"/model/checkpoint.ckpt", global_step=1)
                best_acc = test_total_acc

            print('Test Loss: %.4f, Test Accuracy: %.4f' % (test_total_loss/50, test_total_acc/50))

LC_sess.close()
