# Copyright 2015 Google Inc. All Rights Reserved.
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

"""A very simple MNIST classifier, modified to display data in TensorBoard.

See extensive documentation for the original model at
http://tensorflow.org/tutorials/mnist/beginners/index.md

See documentation on the TensorBoard specific pieces at
http://tensorflow.org/how_tos/summaries_and_tensorboard/index.md

If you modify this file, please update the exerpt in
how_tos/summaries_and_tensorboard/index.md.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow.python.platform
from tensorflow.examples.tutorials.mnist import input_data
# import input_data
import tensorflow as tf
import numbers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils
from sklearn.preprocessing import StandardScaler

import numpy as np

# Import data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


def lselu_3(x):
    with ops.name_scope('lselu')as scope:
        a = 1.0201037102088002
        b = 2.0481567227501563
        x = ops.convert_to_tensor(x)
        ones = array_ops.ones_like(x, x.dtype)
        zeros = array_ops.zeros_like(x, x.dtype)
        mask_0 = tf.where(x >= 0.0, ones, zeros)
        mask_1 = tf.where((x >= -1.5) & (x < 0), ones, zeros)
        mask_2 = tf.where((x >= -3) & (x < -1.5), ones, zeros)
        mask_3 = tf.where(x < -3, ones, zeros)
        y_positive = mask_0 * x
        y_negative = mask_1 * 0.5179 * x + mask_2 * (-0.6035 + 0.1156 * x) + mask_3 * (-0.950213)
        y = a * y_positive + a * b * y_negative
    return  y




def dropout_selu(x, rate, alpha=-1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1 - binary_tensor)

        a = tf.sqrt(fixedPointVar / (keep_prob * ((1 - keep_prob) * tf.pow(alpha - fixedPointMean, 2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
                                lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
                                lambda: array_ops.identity(x))


scaler = StandardScaler().fit(mnist.train.images)
# Parameters
learning_rate = 0.025
training_epochs = 2000  # 一个epoch相当于390次iterations
training_iters = 2000
batch_size = 128
display_step = 1
# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
keep_prob_ReLU = 0.5  # Dropout, probability to keep units
dropout_prob_SNN = 0.05  # Dropout, probability to dropout units
dropout_prob_LSNN = 0.05

# Create the model
x = tf.placeholder(tf.float32, [None, 784], name='n_input')
y = tf.placeholder(tf.float32, [None, 10], name='n_classes')
keep_prob = tf.placeholder(tf.float32)
dropout_prob = tf.placeholder(tf.float32)  # dropout (dropout probability for SNN)
is_training = tf.placeholder(tf.bool)


def _activation_summary(x, name=None):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    _mean, _variance = tf.nn.moments(x, axes=[0])

    tf.summary.histogram(name + '/activations', x)
    # tf.summary.scalar(tensor_name + '/sparsity',
    #                                    tf.nn.zero_fraction(x))
    tf.summary.histogram(name + '-mean', _mean)
    tf.summary.histogram(name + '-variance', _variance)
    tf.summary.scalar(name + '/node1_mean', _mean[0])
    tf.summary.scalar(name + '/node1_variance', _variance[0])
    tf.summary.scalar(name + '/all_mean', tf.reduce_mean(_mean))
    tf.summary.scalar(name + '/all_variance', tf.reduce_mean(_variance))


def weight_variable(shape):
    initial = tf.get_variable('weight', shape=shape,
                              initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / shape[0])))

    return initial


def bias_variable(shape):
    initial = tf.get_variable('bias', shape=shape, initializer=tf.random_normal_initializer(stddev=0))
    # initial = tf.get_variable('bias', shape=shape, initializer=tf.zeros_initializer())

    return initial


def multilayer_perceptron(x,dropout_prob,is_training):
    for i in range(32):
        with tf.variable_scope('layer_%d' % i)as scope:
            weights = weight_variable([784, 784])
            bias = bias_variable([784])
            x = tf.add(tf.matmul(x, weights), bias)
            _activation_summary(x, name='Wx+b')
            x=lselu_3(x)
            _activation_summary(x, name='lselu')
            x = dropout_selu(x, dropout_prob, training=is_training)

            # x = selu(x)
            # _activation_summary(x, name='selu')
            # x = dropout_selu(x, dropout_prob, training=is_training)

            # x=tf.layers.batch_normalization(x,training=is_training)
            # x = tf.layers.batch_normalization(x, axis=-1, training=is_training)  # 指定的channel的维度
            # x = tf.nn.relu(x, name='relu')
            # _activation_summary(x, name='relu')
            # x=tf.nn.relu(x,name='relu')
            # _activation_summary(x, name='relu')
            # x = tf.nn.dropout(x, keep_prob)

    with tf.variable_scope('outlayer')as scope:
        w_outlayer = weight_variable([784, 10])
        b_outlayer = bias_variable([10])
        outlayer = tf.add(tf.matmul(x, w_outlayer), b_outlayer)
        _activation_summary(x, name='Wx+b')
    return outlayer


logits = multilayer_perceptron(x,dropout_prob,is_training)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
cost = tf.reduce_mean(cross_entropy)
loss_tr_op = tf.summary.scalar("train_loss", cost)
loss_te_op = tf.summary.scalar("test_loss", cost)

# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)#很重要！！！！！在训练阶段吧需要更新的统计特征变量手动添加到学习变量中去。
# with tf.control_dependencies(update_ops):
#     train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = 1-tf.reduce_mean(tf.cast(correct_prediction, "float"))
acc_tr_op = tf.summary.scalar("train_error", accuracy)
acc_te_op = tf.summary.scalar("test_error", accuracy)

init = tf.global_variables_initializer()
smry_tr = tf.summary.merge([acc_tr_op, loss_tr_op])  # 这使得对激活值的统计没有统计进去。现在只关注accuracy和loss的统计
smry_va = tf.summary.merge([acc_te_op, loss_te_op])

gpu_options = tf.GPUOptions(allow_growth=True)#按需分配GPU资源
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('MNIST_EPOCH/spelu4_%d' % 32, sess.graph)

    # epoch版本
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x_norm = scaler.transform(batch_x)
            # Run optimization op (backprop)


            sess.run(train_step, feed_dict={x: batch_x_norm, y: batch_y,dropout_prob: dropout_prob_SNN, is_training: True})
            if epoch % display_step == 0:
                loss_LSELU, acc_LSELU, summary_tr = sess.run([cost, accuracy, smry_tr], feed_dict={x: batch_x_norm,
                                                                                                       y: batch_y,
                                                                                                   dropout_prob: 0.0,
                                                                                                 is_training: False})
                writer.add_summary(summary_tr, epoch)

                msg = "epoch {0:>6}:" \
                  "accuracy LSELU: {1:>6.1%} " \
                  "loss  LSELU: {2:.2f} "
                print(msg.format(epoch + 1, acc_LSELU, loss_LSELU))

                acc_LSELU_test, summary_te = sess.run([accuracy, smry_va],
                                                 feed_dict={x: scaler.transform(mnist.test.images[:1000]),
                                                            y: mnist.test.labels[:1000],
                                                            dropout_prob: 0.0, is_training: False})
                writer.add_summary(summary_te, epoch)
                mgs = "Accuracy on Test-Set LSELU:{0:>6.1%} "
                print(mgs.format(acc_LSELU_test))

    # iteration版本
    # step = 0
    # while step < training_iters:
    #     batch_x, batch_y = mnist.train.next_batch(batch_size)
    #     batch_x_norm = scaler.transform(batch_x)
    #     # Run optimization op (backprop)
    #
    #
    #     sess.run(train_step, feed_dict={x: batch_x_norm, y: batch_y,
    #                                     dropout_prob: dropout_prob_SNN, is_training: True})
    #     if step % display_step == 0:
    #         loss_LSELU, acc_LSELU, summary_tr = sess.run([cost, accuracy, smry_tr], feed_dict={x: batch_x_norm,
    #                                                                                          y: batch_y,
    #                                                                                          dropout_prob: 0.0,
    #                                                                                          is_training: False})
    #         writer.add_summary(summary_tr, step)
    #
    #         msg = "step {0:>6}:" \
    #               "accuracy LSELU_3: {1:>6.1%} " \
    #               "loss  LSELU_3: {2:.2f} "
    #         print(msg.format(step + 1, acc_LSELU, loss_LSELU))
    #
    #         acc_LSELU_test, summary_te = sess.run([accuracy, smry_va],
    #                                              feed_dict={x: scaler.transform(mnist.test.images[:1000]),
    #                                                         y: mnist.test.labels[:1000],
    #                                                         dropout_prob: 0.0, is_training: False})
    #         writer.add_summary(summary_te, step)
    #         mgs = "Accuracy on Test-Set LSELU_3:{0:>6.1%} "
    #         print(mgs.format(acc_LSELU_test))
    #
    #     step += 1












