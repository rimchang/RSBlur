import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

if sys.version_info.major == 3:
    xrange = range


def im2uint8(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)


def ResnetBlock(x, dim, ksize, scope='rb'):
    with tf.variable_scope(scope):
        net = slim.conv2d(x, dim, [ksize, ksize], scope='conv1')
        net = slim.conv2d(net, dim, [ksize, ksize], activation_fn=None, scope='conv2')
        return net + x

def polynomal_decay_warm_start(learning_rate, global_step, max_steps, warm_step=10000.0, init_value=0.00001):

    global_step = tf.cast(global_step, tf.float32)

    diff_lr = learning_rate - init_value
    t = global_step / warm_step
    t = tf.minimum(1.0, t)

    lr = init_value + diff_lr * t
    new_step = tf.maximum(global_step-warm_step, 0.0)
    lr = tf.train.polynomial_decay(lr, new_step, max_steps-warm_step, end_learning_rate=0.0,
                              power=0.3)
    return lr