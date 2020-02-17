# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
      inputs=inputs, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size):

  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer())


def building_block(inputs, filters, is_training, projection_shortcut, strides):

    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides)

    inputs = batch_norm_relu(inputs, is_training)
    inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1)

    return inputs + shortcut

def LC_block(inputs, name):
    inputs = conv2d_fixed_padding(inputs=inputs, filters=128, kernel_size=3, strides=1)
    inputs = tf.nn.relu(inputs)
    #inputs = conv2d_fixed_padding(inputs=inputs, filters=128, kernel_size=3, strides=1)
    #inputs = tf.nn.relu(inputs)
    return tf.identity(inputs, name)

def bottleneck_block(inputs, filters, is_training, projection_shortcut, strides):

  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1)

  inputs = batch_norm_relu(inputs, is_training)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides)

  inputs = batch_norm_relu(inputs, is_training)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1)

  return inputs + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides, is_training, name):
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = 4 * filters if block_fn is bottleneck_block else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, is_training, None, 1)

  return tf.identity(inputs, name)


def cifar10_resnet_v2_generator(resnet_size, num_classes):
  if resnet_size % 6 != 2:
    raise ValueError('resnet_size must be 6n + 2:', resnet_size)

  num_blocks = (resnet_size - 2) // 6

  def model(inputs, is_training):
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=16, kernel_size=3, strides=1)
    inputs = tf.identity(inputs, 'initial_conv')

    inputs = block_layer(
        inputs=inputs, filters=16, block_fn=building_block, blocks=num_blocks,
        strides=1, is_training=is_training, name='block_layer1')
    inputs = block_layer(
        inputs=inputs, filters=32, block_fn=building_block, blocks=num_blocks,
        strides=2, is_training=is_training, name='block_layer2')
    inputs = block_layer(
        inputs=inputs, filters=64, block_fn=building_block, blocks=num_blocks,
        strides=2, is_training=is_training, name='block_layer3')

    inputs = batch_norm_relu(inputs, is_training)
    inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=8, strides=1, padding='VALID')
    inputs = tf.identity(inputs, 'final_avg_pool')
    inputs = tf.reshape(inputs, [-1, 64])
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'final_dense')
    return inputs

  return model


def imagenet_resnet_v2_generator(block_fn, layers, num_classes):


  def model(inputs, is_training):

    with tf.variable_scope('BLOCK_1'):
        inputs = conv2d_fixed_padding(inputs=inputs, filters=64, kernel_size=7, strides=2)
        inputs = tf.identity(inputs, 'initial_conv')
        inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=3, strides=2, padding='SAME')
        inputs = tf.identity(inputs, 'initial_max_pool')
        inputs = block_layer(inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
            strides=1, is_training=is_training, name='block_layer1')
    with tf.variable_scope('LC_BLOCK_1'):
        LC_1_output = LC_block(inputs=inputs, pool_size=28, name='LC_block_1')
    with tf.variable_scope('BLOCK_2'):
        inputs = block_layer(inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
            strides=2, is_training=is_training, name='block_layer2')
    with tf.variable_scope('LC_BLOCK_2'):
        LC_2_output = LC_block(inputs=inputs, pool_size=14, name='LC_block_2')
    with tf.variable_scope('BLOCK_3'):
        inputs = block_layer(inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
            strides=2, is_training=is_training, name='block_layer3')
    with tf.variable_scope('LC_BLOCK_3'):
        LC_3_output = LC_block(inputs=inputs, pool_size=7, name='LC_block_3')
    with tf.variable_scope('BLOCK_4'):
        inputs = block_layer(inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
            strides=2, is_training=is_training, name='block_layer4')

        inputs = batch_norm_relu(inputs, is_training)
        inputs = tf.layers.average_pooling2d(inputs=inputs, pool_size=7, strides=1, padding='VALID')
        inputs = tf.identity(inputs, 'final_avg_pool')
        inputs = tf.reshape(inputs, [-1, 512 if block_fn is building_block else 2048])
        inputs = tf.layers.dense(inputs=inputs, units=num_classes)
        inputs = tf.identity(inputs, 'final_dense')
    return inputs, LC_1_output, LC_2_output, LC_3_output

  return model


def imagenet_resnet_v2(resnet_size, num_classes):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      18: {'block': building_block, 'layers': [2, 2, 2, 2]},
      34: {'block': building_block, 'layers': [3, 4, 6, 3]},
      50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
      101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
      152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
      200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
  }

  if resnet_size not in model_params:
    raise ValueError('Not a valid resnet_size:', resnet_size)

  #여기에 직접 layer structure 받을 수 있도록 함수 input도 고치자. 기본은 None으로 하고.

  params = model_params[resnet_size]
  return imagenet_resnet_v2_generator(
      params['block'], params['layers'], num_classes)
