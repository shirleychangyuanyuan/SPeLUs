#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import os

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *


import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
import argparse
import sys
from sklearn.preprocessing import StandardScaler
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

BATCH_SIZE = 100
NUM_UNITS = None


class Model(ModelDesc):

    def __init__(self):
        super(Model, self).__init__()



    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 100.0-1

        image = tf.transpose(image, [0, 3, 1, 2])

        stack1_prob_input = 1.0
        stack2_prob_input = 0.9
        stack3_prob_input = 0.8
        stack4_prob_input = 0.7
        stack5_prob_input = 0.6
        stack6_prob_input = 0.5
        stack7_prob_input = 1.0

        def lselu_3(x,name='lselu3'):
            with ops.name_scope(name)as scope:
                # a = 1.006459281008628
                # b = 2.504714583613515
                a=1.006
                b=2.505
                x = ops.convert_to_tensor(x)
                ones = array_ops.ones_like(x, x.dtype)
                zeros = array_ops.zeros_like(x, x.dtype)
                mask_0 = tf.where(x >= 0.0, ones, zeros)
                mask_1 = tf.where((x >= -2.2) & (x < 0), ones, zeros)
                mask_2 = tf.where(x < -2.2, ones, zeros)
                y_positive = mask_0 * x
                y_negative = mask_1 * 0.4042 * x + mask_2 * (-0.889197)
                y = a * y_positive + a * b * y_negative
            return y

        def elu_network(x):
            # stack1
            conv1_stack1 = Conv2D('stack1/conv1', x, out_channel=32, kernel_shape=5, padding='SAME', stride=1,
                                  W_init=tf.random_normal_initializer(stddev=np.sqrt(1 / (5 * 5 * 3))), nl=tf.identity)
            #add_activation_summary(conv1_stack1,name='layer1/pre_activation')
            conv1_stack1 = lselu_3(conv1_stack1)
            #add_activation_summary(conv1_stack1,name='layer1/after_activation')
            stack1_dropout = Dropout(conv1_stack1, keep_prob=stack1_prob_input)
            pool1 = MaxPooling('stack1/pool', stack1_dropout, shape=2, stride=2, padding='SAME')

            # stack2
            conv1_stack2 = Conv2D('stack2/conv1', pool1, out_channel=64, kernel_shape=3, padding='SAME', stride=1,
                                  W_init=tf.random_normal_initializer(stddev=np.sqrt(1 / (9*32))), nl=tf.identity)
            #add_activation_summary(conv1_stack2, name='layer2/pre_activation')
            conv1_stack2 = lselu_3(conv1_stack2)
            #add_activation_summary(conv1_stack2,name='layer2/after_activation')
            conv2_stack2 = Conv2D('stack2/conv2', conv1_stack2, out_channel=64, kernel_shape=3, padding='SAME',
                                  stride=1,W_init=tf.random_normal_initializer(stddev=np.sqrt(1 / (9 * 64))), nl=tf.identity)
            #add_activation_summary(conv2_stack2, name='layer3/pre_activation')
            conv2_stack2 = lselu_3(conv2_stack2)
            #add_activation_summary(conv2_stack2,name='layer3/after_activation')
            stack2_dropout = Dropout(conv2_stack2, keep_prob=stack2_prob_input)
            pool2 = MaxPooling('stack2/pool', stack2_dropout, shape=2, stride=2, padding='SAME')

            # satck3
            conv1_stack3 = Conv2D('stack3/conv1', pool2, out_channel=128, kernel_shape=3, padding='SAME',
                                  stride=1, W_init=tf.random_normal_initializer(stddev=np.sqrt(1 / (9*64))), nl=tf.identity)
            #add_activation_summary(conv1_stack3, name='layer4/pre_activation')
            conv1_stack3 = lselu_3(conv1_stack3)
            #add_activation_summary(conv1_stack3,name='layer4/after_activation')
            conv2_stack3 = Conv2D('stack3/conv2', conv1_stack3, out_channel=128, kernel_shape=2, padding='SAME',
                                  stride=1, W_init=tf.random_normal_initializer(stddev=np.sqrt(1 / (4 * 128))),nl=tf.identity)
            #add_activation_summary(conv2_stack3, name='layer5/pre_activation')
            conv2_stack3 = lselu_3(conv2_stack3)
            #add_activation_summary(conv2_stack3,name='layer5/after_activation')
            stack3_dropout = Dropout(conv2_stack3, keep_prob=stack3_prob_input)
            pool3 = MaxPooling('stack3/pool', stack3_dropout, shape=2, stride=2, padding='SAME')

            # satck4
            conv1_stack4 = Conv2D('stack4/conv1', pool3, out_channel=256, kernel_shape=3, padding='SAME', stride=1,
                                  W_init=tf.random_normal_initializer(stddev=np.sqrt(1 / (9*128))), nl=tf.identity)
            #add_activation_summary(conv1_stack4, name='layer6/pre_activation')
            conv1_stack4 = lselu_3(conv1_stack4)
            #add_activation_summary(conv1_stack4,name='layer6/after_activation')
            conv2_stack4 = Conv2D('stack4/conv2', conv1_stack4, out_channel=256, kernel_shape=2, padding='SAME',
                                  stride=1,W_init=tf.random_normal_initializer(stddev=np.sqrt(1 / (4 * 256))), nl=tf.identity)
            #add_activation_summary(conv2_stack4, name='layer7/pre_activation')
            conv2_stack4 = lselu_3(conv2_stack4)
            #add_activation_summary(conv2_stack4,name='layer7/after_activation')
            stack4_dropout = Dropout(conv2_stack4, keep_prob=stack4_prob_input)
            pool4 = MaxPooling('stack4/pool', stack4_dropout, shape=2, stride=2, padding='SAME')

            # stack5
            conv1_stack5 = Conv2D('stack5/conv1', pool4, out_channel=280, kernel_shape=3, padding='SAME', stride=1,
                                  W_init=tf.random_normal_initializer(stddev=np.sqrt(1 / (9*256))), nl=tf.identity)
           # add_activation_summary(conv1_stack5, name='layer8/pre_activation')
            conv1_stack5 = lselu_3(conv1_stack5)
            #add_activation_summary(conv1_stack5,name='layer8/after_activation')
            conv2_stack5 = Conv2D('stack5/conv2', conv1_stack5, out_channel=280, kernel_shape=2, padding='SAME',
                                  stride=1,W_init=tf.random_normal_initializer(stddev=np.sqrt(1 / (4 * 280))), nl=tf.identity)
            #add_activation_summary(conv2_stack5, name='layer9/pre_activation')
            conv2_stack5 = lselu_3(conv2_stack5)
            #add_activation_summary(conv2_stack5,name='layer9/after_activation')
            stack5_dropout = Dropout(conv2_stack5, keep_prob=stack5_prob_input)
            pool5 = MaxPooling('stack5/pool', stack5_dropout, shape=2, stride=2, padding='SAME')

            # stack6
            conv1_stack6 = Conv2D('stack6/conv1', pool5, out_channel=300, kernel_shape=3, padding='SAME', stride=1,
                                  W_init=tf.random_normal_initializer(stddev=np.sqrt(1 / (9*280))), nl=tf.identity)
            #add_activation_summary(conv1_stack6, name='layer10/pre_activation')
            conv1_stack6 = lselu_3(conv1_stack6)
            #add_activation_summary(conv1_stack6,name='layer10/after_activation')
            stack6_dropout = Dropout(conv1_stack6, keep_prob=stack6_prob_input)

            # stack7
            conv1_stack7 = Conv2D('stack7/conv1', stack6_dropout, out_channel=100, kernel_shape=1, padding='SAME',
                                  stride=1,W_init=tf.random_normal_initializer(stddev=np.sqrt(1 / 300)), nl=tf.identity)
            #add_activation_summary(conv1_stack7,name='layer11/pre_activation')
            conv1_stack7 = lselu_3(conv1_stack7)
            #add_activation_summary(conv1_stack7, name='layer11/after_activation')
            stack7_dropout = Dropout(conv1_stack7, keep_prob=stack7_prob_input)

            y_conv_reshape = tf.reshape(stack7_dropout, (-1, 100))
            return y_conv_reshape



        with argscope([Conv2D, MaxPooling], data_format='NCHW'):
            logits=elu_network(image)


        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='train_loss')
        #add_activation_summary(cost,name='train_loss')

        wrong = prediction_incorrect(logits, label)

        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_w = 0.0005
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost,wd_cost)

        add_param_summary(('.*/W', ['histogram']))  # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 1e-2, summary=True)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt


def get_data(train_or_test,cifar_classnum):
    isTrain = train_or_test == 'train'
    if cifar_classnum==10:
        ds = dataset.Cifar10(train_or_test)
    else:
        ds = dataset.Cifar100(train_or_test)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean),#减去所有像素平均值
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds


def get_config():
    mod = sys.modules['__main__']
    basename = os.path.basename(mod.__file__)

    logdir = os.path.join('train_log',
                          'cifar100-spelu3-simple'
                           )

    logger.set_logger_dir(logdir)

    import shutil
    # import pdb; pdb.set_trace()

    shutil.copy(mod.__file__, logger.LOG_DIR)

    dataset_train = get_data('train',100)
    dataset_test = get_data('test',100)

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            #ModelSaver(),不保存每一次的model
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError()]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 1e-2), (70,0.005), (100,0.0005), (130, 0.00005)])

        ],
        model=Model(),
        max_epoch=150,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')  # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    parser.add_argument("-g", "--gpuid", type=str, help='GPU to use (leave blank for CPU only)', default="1")
   # parser.add_argument('--n', help='Nsize', type=int, default=3)
    parser.add_argument('--c', help='cardinality', type=int, default=1)
    parser.add_argument('--t', help='hyperparameter t', type=float, default=0.05)
    args = parser.parse_args()

    #if args.gpu:
     #   os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid



    config = get_config()
    gpu_options = tf.GPUOptions(allow_growth=True)
    config.session_config=tf.ConfigProto(gpu_options=gpu_options)


    if args.load:
        config.session_init = SaverRestore(args.load)
    config.nr_tower = max(get_nr_gpu(), 1)
    SyncMultiGPUTrainer(config).train()
