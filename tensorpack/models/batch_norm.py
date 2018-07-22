#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: batch_norm.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from tensorflow.contrib.framework import add_model_variable
from tensorflow.python.training import moving_averages

from ..tfutils.tower import get_current_tower_context
from ..utils import logger
from .common import layer_register, VariableHolder

__all__ = ['BatchNorm', 'BatchRenorm']

# decay: being too close to 1 leads to slow start-up. torch use 0.9.
# eps: torch: 1e-5. Lasagne: 1e-4


def get_bn_variables(n_out, use_scale, use_bias, gamma_init):
    if use_bias:
        beta = tf.get_variable('beta', [n_out], initializer=tf.constant_initializer())
    else:
        beta = tf.zeros([n_out], name='beta')
    if use_scale:
        gamma = tf.get_variable('gamma', [n_out], initializer=gamma_init)
    else:
        gamma = tf.ones([n_out], name='gamma')
    # x * gamma + beta

    moving_mean = tf.get_variable('mean/EMA', [n_out],
                                  initializer=tf.constant_initializer(), trainable=False)
    moving_var = tf.get_variable('variance/EMA', [n_out],
                                 initializer=tf.constant_initializer(), trainable=False)
    return beta, gamma, moving_mean, moving_var


def update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, decay):#在训练的过程中，通过每个step得到的mean和variance，
                                                                     # 叠加计算对应的moving_average（滑动平均），并最终保存下来以便在inference的过程中使用。
    # TODO is there a way to use zero_debias in multi-GPU?
    update_op1 = moving_averages.assign_moving_average(
        moving_mean, batch_mean, decay, zero_debias=False,
        name='mean_ema_op')
    update_op2 = moving_averages.assign_moving_average(
        moving_var, batch_var, decay, zero_debias=False,
        name='var_ema_op')
    # Only add model var when we update them
    add_model_variable(moving_mean)
    add_model_variable(moving_var)

    # TODO add an option, and maybe enable it for replica mode?
    # with tf.control_dependencies([update_op1, update_op2]):
    # return tf.identity(xn, name='output')
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op1)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op2)
    return xn


def reshape_for_bn(param, ndims, chan, data_format):
    if ndims == 2:
        shape = [1, chan]
    else:
        shape = [1, 1, 1, chan] if data_format == 'NHWC' else [1, chan, 1, 1]
    return tf.reshape(param, shape)


@layer_register(log_shape=False)
def BatchNorm(x, use_local_stat=None, decay=0.9, epsilon=1e-5,
              use_scale=True, use_bias=True,
              gamma_init=tf.constant_initializer(1.0), data_format='NHWC'):
    """
    Batch Normalization layer, as described in the paper:
    `Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariance Shift <http://arxiv.org/abs/1502.03167>`_.

    Args:
        x (tf.Tensor): a 4D or 2D tensor. When 4D, the layout should match data_format.
        use_local_stat (bool): whether to use mean/var of the current batch or the moving average.
            Defaults to True in training and False in inference.
        decay (float): decay rate of moving average.
        epsilon (float): epsilon to avoid divide-by-zero.
        use_scale, use_bias (bool): whether to use the extra affine transformation or not.
        gamma_init: initializer for gamma (the scale).

    Returns:
        tf.Tensor: a tensor named ``output`` with the same shape of x.

    Variable Names:

    * ``beta``: the bias term. Will be zero-inited by default.
    * ``gamma``: the scale term. Will be one-inited by default.
        Input will be transformed by ``x * gamma + beta``.
    * ``mean/EMA``: the moving average of mean.
    * ``variance/EMA``: the moving average of variance.

    Note:
        In multi-GPU training, moving averages across GPUs are not aggregated.
        This is consistent with most frameworks.

        However, all GPUs use the moving averages on the first GPU (instead of
        their own), this is inconsistent with most frameworks (but consistent
        with the official inceptionv3 example).
    """
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4]
    if ndims == 2:
        data_format = 'NHWC'
    if data_format == 'NCHW':
        n_out = shape[1]
    else:
        n_out = shape[-1]  # channel
    assert n_out is not None, "Input to BatchNorm cannot have unknown channels!"
    beta, gamma, moving_mean, moving_var = get_bn_variables(n_out, use_scale, use_bias, gamma_init)#建立一些变量，前两个需要训练，后两个不训练

    ctx = get_current_tower_context()
    if use_local_stat is None:
        use_local_stat = ctx.is_training#在训练阶段的话就是True
    elif use_local_stat != ctx.is_training:
        # we allow the use of local_stat in testing (only print warnings)
        # because it is useful to certain applications.
        logger.warn("[BatchNorm] use_local_stat != is_training")

    if use_local_stat:#在训练阶段
        if ndims == 2:
            x = tf.reshape(x, [-1, 1, 1, n_out])    # fused_bn only takes 4D input
            # fused_bn has error using NCHW? (see #190)

        xn, batch_mean, batch_var = tf.nn.fused_batch_norm(
            x, gamma, beta, epsilon=epsilon,
            is_training=True, data_format=data_format)

        if ndims == 2:
            xn = tf.squeeze(xn, [1, 2])#删除第一维度和第二维度的1
    else:#在测试阶段
        assert not ctx.is_training, "In training, local statistics has to be used!"
        # non-fused op is faster for inference
        if ndims == 4 and data_format == 'NCHW':
            [g, b, mm, mv] = [reshape_for_bn(_, ndims, n_out, data_format)
                              for _ in [gamma, beta, moving_mean, moving_var]]
            xn = tf.nn.batch_normalization(x, mm, mv, b, g, epsilon)#用的是moving mean和meaning variance
        else:
            # avoid the reshape if possible (when channel is the last dimension)
            xn = tf.nn.batch_normalization(
                x, moving_mean, moving_var, beta, gamma, epsilon)

    # maintain EMA only on one GPU is OK.
    if ctx.is_main_training_tower:
        ret = update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, decay)
    else:
        ret = tf.identity(xn, name='output')

    vh = ret.variables = VariableHolder(mean=moving_mean, variance=moving_var)
    if use_scale:
        vh.gamma = gamma
    if use_bias:
        vh.beta = beta
    return ret


@layer_register(log_shape=False)
def BatchRenorm(x, rmax, dmax, decay=0.9, epsilon=1e-5,
                use_scale=True, use_bias=True, data_format='NHWC'):
    """
    Batch Renormalization layer, as described in the paper:
    `Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models
    <https://arxiv.org/abs/1702.03275>`_.

    Args:
        x (tf.Tensor): a NHWC or NC tensor.
        rmax, dmax (tf.Tensor): a scalar tensor, the maximum allowed corrections.
        decay (float): decay rate of moving average.
        epsilon (float): epsilon to avoid divide-by-zero.
        use_scale, use_bias (bool): whether to use the extra affine transformation or not.

    Returns:
        tf.Tensor: a tensor named ``output`` with the same shape of x.

    Variable Names:

    * ``beta``: the bias term.
    * ``gamma``: the scale term. Input will be transformed by ``x * gamma + beta``.
    * ``mean/EMA``: the moving average of mean.
    * ``variance/EMA``: the moving average of variance.
    """

    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4]
    if ndims == 2:
        data_format = 'NHWC'    # error using NCHW? (see #190)
    if data_format == 'NCHW':
        n_out = shape[1]
    else:
        n_out = shape[-1]  # channel
    assert n_out is not None, "Input to BatchRenorm cannot have unknown channels!"

    beta, gamma, moving_mean, moving_var = get_bn_variables(
        n_out, use_scale, use_bias, tf.constant_initializer(1.0))

    ctx = get_current_tower_context()
    use_local_stat = ctx.is_training
    # for BatchRenorm, use_local_stat should always be is_training, unless a
    # different usage comes out in the future.

    if use_local_stat:
        if ndims == 2:
            x = tf.reshape(x, [-1, 1, 1, n_out])

        xn, batch_mean, batch_var = tf.nn.fused_batch_norm(
            x, gamma, beta, epsilon=epsilon, is_training=True, data_format=data_format)

        inv_sigma = tf.rsqrt(moving_var, 'inv_sigma')
        r = tf.stop_gradient(tf.clip_by_value(
            tf.sqrt(batch_var) * inv_sigma, 1.0 / rmax, rmax))
        d = tf.stop_gradient(tf.clip_by_value(
            (batch_mean - moving_mean) * inv_sigma,
            -dmax, dmax))
        r = reshape_for_bn(r, ndims, n_out, data_format)
        d = reshape_for_bn(d, ndims, n_out, data_format)
        xn = xn * r + d

        if ndims == 2:
            xn = tf.squeeze(xn, [1, 2])

    else:
        if ndims == 4 and data_format == 'NCHW':
            [g, b, mm, mv] = [reshape_for_bn(_, ndims, n_out, data_format)
                              for _ in [gamma, beta, moving_mean, moving_var]]
            xn = tf.nn.batch_normalization(x, mm, mv, b, g, epsilon)
        else:
            xn = tf.nn.batch_normalization(
                x, moving_mean, moving_var, beta, gamma, epsilon)

    # training also needs EMA, so ideally we should maintain it on every tower
    if ctx.is_main_training_tower or ctx.has_own_variables:
        ret = update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, decay)
    else:
        ret = tf.identity(xn, name='output')

    vh = ret.variables = VariableHolder(mean=moving_mean, variance=moving_var)
    if use_scale:
        vh.gamma = gamma
    if use_bias:
        vh.beta = beta
    return ret
