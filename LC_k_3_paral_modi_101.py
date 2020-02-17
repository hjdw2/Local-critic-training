import random
import tensorflow as tf

from tqdm import tqdm # Used to display training progress bar
import os
import numpy as np
import sys
import time
from data_utility import *
from resnet_model import *
sys.path.append(os.pardir)

cwd = os.getcwd()
cwd = cwd.replace("\\", "/")
tf.set_random_seed(1)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class_num = 10
train_number = 50000
test_number = 10000
weight_decay = 0.0005

iterations = 80000
batch_size = 128 # modified to evenly divide dataset size

validation_checkpoint = 800 # How often (iterations) to validate model

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sg_sess = tf.Session(config=config)

train_x, train_y, test_x, test_y = prepare_data()
train_x, test_x = data_preprocessing(train_x, test_x)

is_training = True
num_classes = 10
block_fn = bottleneck_block
layers = [3, 4, 23, 3]

# Ops for network architecture
with tf.variable_scope("architecture"):
    # Inputs
    with tf.variable_scope("input"):
        X = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name="data")
        Y = tf.placeholder(tf.float32, shape=(None, class_num), name="labels") # Target
        learning_rate =  tf.placeholder(tf.float32, shape =[])
    '''
    with tf.device("/gpu:0"):
        with tf.variable_scope("res_block_1"):
            h2 = conv_layer(X, 64, 3, 1,training=True, name="layer1")
            h2_1 = conv_layer(h2, 64, 3, 1,training=True,name="layer2")
        with tf.variable_scope("sg_block_1"):
            h_sg1= conv_layer_sg(h2_1, 128, 3, 1, name="sg1_1")
            h_sg1 = dense_layer(h_sg1, name="sg1_2")

    with tf.device("/gpu:1"):
        with tf.variable_scope("res_block_2"):
            h3 = conv_layer(h2_1, 128, 3, 2,training=True, name="layer3")
            h3 = conv_layer(h3, 128, 3, 1, training=True,name="layer4")
            h3_1 = conv_layer(h3, 128, 3, 1, training=True,name="layer5")
        with tf.variable_scope("sg_block_2"):
            h_sg2= conv_layer_sg(h3_1, 128, 3, 1, name="sg2_1")
            h_sg2 = dense_layer(h_sg2, name="sg2_2")

    with tf.device("/gpu:2"):
        with tf.variable_scope("res_block_3"):
            h4 = conv_layer(h3_1, 256, 3, 2, training=True,name="layer6")
            h4 = conv_layer(h4, 256, 3, 1,training=True, name="layer7")
            h4_1 = conv_layer(h4, 256, 3, 1, training=True,name="layer8")
        with tf.variable_scope("sg_block_3"):
            h_sg3= conv_layer_sg(h4_1, 128, 3, 1, name="sg3_1")
            h_sg3 = dense_layer(h_sg3, name="sg3_2")

    with tf.device("/gpu:3"):
        with tf.variable_scope("res_block_4"):
            h5 = conv_layer(h4_1, 512, 3, 2, training=True,name="layer9")
            h5 = conv_layer(h5, 512, 3, 1, training=True,name="layer10")
            h5 = conv_layer(h5, 512, 3, 1,training=True, name="layer11")
            logits = dense_layer(h5, name="layer12")

    '''
    with tf.device("/gpu:0"):
        with tf.variable_scope('res_block_1'):
            h2 = conv2d_fixed_padding(inputs=X, filters=64, kernel_size=3, strides=1)
            h2 = block_layer(inputs=h2, filters=64, block_fn=block_fn, blocks=layers[0], strides=1, is_training=is_training, name='block_layer1')
            h2_1 = block_layer(inputs=h2, filters=128, block_fn=block_fn, blocks=layers[1], strides=2, is_training=is_training, name='block_layer2')

    with tf.device("/gpu:1"):
        with tf.variable_scope('res_block_2'):
            h3_1 = block_layer(inputs=h2_1, filters=256, block_fn=block_fn, blocks=8, strides=2, is_training=is_training, name='block_layer3')

    with tf.device("/gpu:2"):
        with tf.variable_scope('res_block_3'):
            h4_1 = block_layer(inputs=h3_1, filters=256, block_fn=block_fn, blocks=8, strides=1, is_training=is_training, name='block_layer4')

    with tf.device("/gpu:3"):
        with tf.variable_scope('res_block_4'):
            h5 = block_layer(inputs=h4_1, filters=256, block_fn=block_fn, blocks=5, strides=1, is_training=is_training, name='block_layer5')
            h5 = block_layer(inputs=h5, filters=512, block_fn=block_fn, blocks=layers[3], strides=2, is_training=is_training, name='block_layer6')
            h5 = batch_norm_relu(h5, is_training)
            logits = dense_layer(inputs=h5, name='logits')

    with tf.device("/gpu:0"):
        with tf.variable_scope('sg_block_1'):
            h_sg1 = LC_block(inputs=h2_1, name='LC_block_1')
            h_sg1 = dense_layer(h_sg1, name="sg1_2")
    with tf.device("/gpu:1"):
        with tf.variable_scope('sg_block_2'):
            h_sg2 = LC_block(inputs=h3_1, name='LC_block_2')
            h_sg2 = dense_layer(h_sg2, name="sg2_2")
    with tf.device("/gpu:2"):
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

def train_layer_n(n, h_n, loss_layer=None):
    if n == 4:
        layer_grads = tf.gradients(h_n + tf.add_n([tf.nn.l2_loss(var) for var in layer_vars[n-1]]) * weight_decay ,layer_vars[n-1])
    else:
        layer_grads = tf.gradients(h_n + tf.add_n([tf.nn.l2_loss(var) for var in layer_vars[n-1]]) * weight_decay ,layer_vars[n-1], tf.gradients(loss_layer, h_n))

    layer_gv = list(zip(layer_grads,layer_vars[n-1]))
    layer_opt = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum = 0.9,use_nesterov=True).apply_gradients(layer_gv)

    return layer_opt

with tf.device("/gpu:0"):
    pred_loss_sg1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=h_sg1, labels=Y)
with tf.device("/gpu:1"):
    pred_loss_sg2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=h_sg2, labels=Y)
with tf.device("/gpu:2"):
    pred_loss_sg3 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=h_sg3, labels=Y)
with tf.device("/gpu:3"):
    pred_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
    print_pred_loss = tf.reduce_mean(pred_loss)

with tf.device("/gpu:3"):
    layer4_opt = train_layer_n(4, tf.reduce_mean(pred_loss))

with tf.device("/gpu:2"):
    sg4_opt = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(tf.reduce_mean(tf.abs(pred_loss - pred_loss_sg3)), var_list=sg_vars[2])
    layer3_opt = train_layer_n(3, h4_1, loss_layer=tf.reduce_mean(pred_loss_sg3))

with tf.device("/gpu:1"):
    sg3_opt = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(tf.reduce_mean(tf.abs(pred_loss_sg3 - pred_loss_sg2)), var_list=sg_vars[1])
    layer2_opt = train_layer_n(2, h3_1, loss_layer=tf.reduce_mean(pred_loss_sg2))

with tf.device("/gpu:0"):
    sg2_opt = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(tf.reduce_mean(tf.abs(pred_loss_sg2 - pred_loss_sg1)), var_list=sg_vars[0])
    layer1_opt = train_layer_n(1, h2_1, loss_layer=tf.reduce_mean(pred_loss_sg1))

with tf.device("/gpu:3"):
    correct_preds = tf.equal(tf.argmax(logits,1), tf.argmax(Y,1), name="correct_predictions")
    accuracy = tf.reduce_sum(tf.cast(correct_preds,tf.int32), name="correct_prediction_count") / batch_size

# Ops for tensorboard summary data
with tf.variable_scope("summary"):
    #cost_summary_opt = tf.summary.scalar("loss", pred_loss)
    #accuracy_summary_opt = tf.summary.scalar("accuracy", accuracy)
    #accuracy1_summary_opt = tf.summary.scalar("accuracy_1", accuracy_1)
    summary_op = tf.summary.merge_all()

# Train using synthetic gradients
with sg_sess.as_default():
    sg_sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver.restore(sg_sess, cwd+"/model/model.ckpt-1125")
    total_time = 0

    f = open(cwd+"/K_3_loss_time.txt", 'w')
    start_time = time.time()
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

        sg_sess.run([layer4_opt, layer3_opt, layer2_opt, layer1_opt, sg2_opt, sg3_opt, sg4_opt, extra_update_ops], feed_dict={X:data,Y:target, learning_rate:lr})

        if i % validation_checkpoint == 0:
            data, target = next_batch(batch_size, train_x, train_y, train_number)
            batch_accuracy_t = sg_sess.run(accuracy, feed_dict={X:data,Y:target})
            print(batch_accuracy_t)

            Xb, Yb = next_batch(batch_size, test_x, test_y, test_number)
            batch_accuracy_b = sg_sess.run(accuracy, feed_dict={X:Xb,Y:Yb})
            print(batch_accuracy_b)

            loss_train = 0
            loss_test = 0
            train_time = time.time() - start_time
            total_time += train_time

            for j in range(0,500):
                loss_train += sg_sess.run(print_pred_loss, feed_dict={X:(train_x[j*100:(j+1)*100,:,:,:]),Y:train_y[j*100:(j+1)*100,:]})

            for j in range(0,100):
                loss_test += sg_sess.run(print_pred_loss, feed_dict={X:(test_x[j*100:(j+1)*100,:,:,:]),Y:test_y[j*100:(j+1)*100,:]})

            start_time = time.time()
            f.write("%.3f %.3f %.3f \n" %(loss_train/500, loss_test/100, total_time))

    save_path = saver.save(sg_sess,cwd+"/model/LC_K_3_paral.ckpt", global_step=1)
    f.close()

sg_sess.close()
