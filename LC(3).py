import random
import tensorflow as tf

from tqdm import tqdm # Used to display training progress bar
import os
import numpy as np
import sys
import time
from data_utility import *
from resnet_model import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
sys.path.append(os.pardir)

cwd = os.getcwd()
cwd = cwd.replace("\\", "/")
cwd = cwd[2:]

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class_num = 10
train_number = 50000
test_number = 10000
weight_decay = 0.0005

iterations = 80000
batch_size = 128 # modified to evenly divide dataset size

validation_checkpoint = 100 # How often (iterations) to validate model

sg_sess = tf.Session()
backprop_sess = tf.Session()

train_x, train_y, test_x, test_y = prepare_data()
train_x, test_x = data_preprocessing(train_x, test_x)

is_training = True
num_classes = 10
block_fn = bottleneck_block
layers = [3, 4, 6, 3]

# Ops for network architecture
with tf.variable_scope("architecture"):
    # Inputs
    with tf.variable_scope("input"):
        X = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name="data")
        Y = tf.placeholder(tf.float32, shape=(None, class_num), name="labels") # Target
        learning_rate =  tf.placeholder(tf.float32, shape =[])

    # Inference layers
    with tf.variable_scope('res_block_1'):
        h2 = conv2d_fixed_padding(inputs=X, filters=64, kernel_size=3, strides=1)
        h2_1 = block_layer(inputs=h2, filters=64, block_fn=block_fn, blocks=layers[0], strides=1, is_training=is_training, name='block_layer1')

    with tf.variable_scope('res_block_2'):
        h3_1 = block_layer(inputs=h2_1, filters=128, block_fn=block_fn, blocks=layers[1], strides=2, is_training=is_training, name='block_layer2')

    with tf.variable_scope('res_block_3'):
        h4_1 = block_layer(inputs=h3_1, filters=256, block_fn=block_fn, blocks=layers[2], strides=2, is_training=is_training, name='block_layer3')

    with tf.variable_scope('res_block_4'):
        h5 = block_layer(inputs=h4_1, filters=512, block_fn=block_fn, blocks=layers[3], strides=2, is_training=is_training, name='block_layer4')
        h5 = batch_norm_relu(h5, is_training)
        logits = dense_layer(inputs=h5, name='logits')

    with tf.variable_scope('sg_block_1'):
        h_sg1 = LC_block(inputs=h2_1, name='LC_block_1')
        h_sg1 = dense_layer(h_sg1, name="sg1_2")
    with tf.variable_scope('sg_block_2'):
        h_sg2 = LC_block(inputs=h3_1, name='LC_block_2')
        h_sg2 = dense_layer(h_sg2, name="sg2_2")
    with tf.variable_scope('sg_block_3'):
        h_sg3 = LC_block(inputs=h4_1, name='LC_block_3')
        h_sg3 = dense_layer(h_sg3, name="sg3_2")

# Collections of trainable variables in each block
layer_vars = [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="architecture/res_block_1"),
tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="architecture/res_block_2"),
tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="architecture/res_block_3"),
tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="architecture/res_block_4")]

sg_vars = [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="architecture/sg_block_1"),
tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="architecture/sg_block_2"),
tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="architecture/sg_block_3")]

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

def train_layer_n(n, h_n, loss_layer=False, loss_target_sg=False, loss_sg=False):
    if n == 4:
        layer_grads = tf.gradients(h_n + tf.add_n([tf.nn.l2_loss(var) for var in layer_vars[n-1] ]) * weight_decay ,layer_vars[n-1])
    else:
        layer_grads = tf.gradients(h_n + tf.add_n([tf.nn.l2_loss(var) for var in layer_vars[n-1] ]) * weight_decay ,layer_vars[n-1], tf.gradients(loss_layer, h_n))

    layer_gv = list(zip(layer_grads,layer_vars[n-1]))
    layer_opt = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum = 0.9,use_nesterov=True).apply_gradients(layer_gv)

    if n == 1:
        return layer_opt
    else:
        sg_loss =  tf.reduce_mean(tf.abs( tf.subtract(loss_target_sg, loss_sg)))
        sg_opt = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(sg_loss, var_list=sg_vars[n-2])
        return layer_opt, sg_opt

# Ops for training
with tf.variable_scope("train"):
    pred_loss =  tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
    pred_loss_sg1 =  tf.nn.softmax_cross_entropy_with_logits_v2(logits=h_sg1, labels=Y)
    pred_loss_sg2 =  tf.nn.softmax_cross_entropy_with_logits_v2(logits=h_sg2, labels=Y)
    pred_loss_sg3 =  tf.nn.softmax_cross_entropy_with_logits_v2(logits=h_sg3, labels=Y)

    # Optimizers when using synthetic gradients
    with tf.variable_scope("synthetic"):
        layer4_opt, sg4_opt = train_layer_n(4,tf.reduce_mean(pred_loss),   loss_target_sg=pred_loss, loss_sg=pred_loss_sg3)
        layer3_opt, sg3_opt = train_layer_n(3, h4_1, loss_layer=tf.reduce_mean(pred_loss_sg3), loss_target_sg=pred_loss_sg3, loss_sg=pred_loss_sg2)
        layer2_opt, sg2_opt = train_layer_n(2, h3_1, loss_layer=tf.reduce_mean(pred_loss_sg2), loss_target_sg=pred_loss_sg2, loss_sg=pred_loss_sg1)
        layer1_opt = train_layer_n(1, h2_1, loss_layer=tf.reduce_mean(pred_loss_sg1))

# Ops for validation and testing (computing classification accuracy)
with tf.variable_scope("test"):
    correct_preds = tf.equal(tf.argmax(logits,1), tf.argmax(Y,1), name="correct_predictions")
    accuracy = tf.reduce_sum(tf.cast(correct_preds,tf.int32), name="correct_prediction_count") / batch_size

    correct_preds_1 = tf.equal(tf.argmax(h_sg1,1), tf.argmax(Y,1), name="correct_predictions_1")
    accuracy_1 = tf.reduce_sum(tf.cast(correct_preds_1,tf.int32), name="correct_prediction_count_1") / batch_size

    correct_preds_2 = tf.equal(tf.argmax(h_sg2,1), tf.argmax(Y,1), name="correct_predictions_2")
    accuracy_2 = tf.reduce_sum(tf.cast(correct_preds_2,tf.int32), name="correct_prediction_count_2") / batch_size

    correct_preds_3 = tf.equal(tf.argmax(h_sg3,1), tf.argmax(Y,1), name="correct_predictions_3")
    accuracy_3 = tf.reduce_sum(tf.cast(correct_preds_3,tf.int32), name="correct_prediction_count_3") / batch_size

    correct_preds_123 = tf.equal(tf.argmax(h_sg3+h_sg2+h_sg1,1), tf.argmax(Y,1), name="correct_predictions_123")
    accuracy_123 = tf.reduce_sum(tf.cast(correct_preds_123,tf.int32), name="correct_prediction_count_123") / batch_size

    correct_preds_1234 = tf.equal(tf.argmax(h_sg3+h_sg2+h_sg1+logits,1), tf.argmax(Y,1), name="correct_predictions_1234")
    accuracy_1234 = tf.reduce_sum(tf.cast(correct_preds_1234,tf.int32), name="correct_prediction_count_1234") / batch_size


# Ops for tensorboard summary data
with tf.variable_scope("summary"):
    #cost_summary_opt = tf.summary.scalar("loss", pred_loss)
    accuracy_summary_opt = tf.summary.scalar("accuracy", accuracy)
    accuracy1_summary_opt = tf.summary.scalar("accuracy_1", accuracy_1)
    accuracy2_summary_opt = tf.summary.scalar("accuracy_2", accuracy_2)
    accuracy3_summary_opt = tf.summary.scalar("accuracy_3", accuracy_3)

    summary_op = tf.summary.merge_all()

# Train using synthetic gradients
with sg_sess.as_default():
    sg_sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver.restore(sg_sess, cwd+"/model/model.ckpt-1125")

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
        # Each layer can now be independently updated (could be parallelized)

        sg_sess.run([layer4_opt, layer3_opt, layer2_opt, layer1_opt, sg2_opt, sg3_opt, sg4_opt, extra_update_ops], feed_dict={X:data,Y:target, learning_rate:lr})

        if i % validation_checkpoint == 0:
            data, target = next_batch(batch_size, train_x, train_y, train_number)
            batch_accuracy_t = sg_sess.run(accuracy, feed_dict={X:data,Y:target})
            print(batch_accuracy_t)

            Xb, Yb = next_batch(batch_size, test_x, test_y, test_number)
            batch_accuracy_b = sg_sess.run(accuracy, feed_dict={X:Xb,Y:Yb})
            print(batch_accuracy_b)

            if (i == 2000) & (batch_accuracy_t < 0.1) :
                sys.exit()

    save_path = saver.save(sg_sess,cwd+"/model/ResNet50_LC.ckpt", global_step=1)


backprop_sess.close()
sg_sess.close()
