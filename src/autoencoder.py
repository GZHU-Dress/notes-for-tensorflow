"""TF 实现自编码器."""
# !/usr/bin/env python
# coding=utf-8


import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def xavier_init(fan_in, fan_out, constant=1):
    """Xaiver 初始化器，自动优化权重."""
    low = -constant * np.sqrt(6 / (fan_in + fan_out))  # fan_in 是输入节点数量
    high = constant * np.sqrt(6 / (fan_in + fan_out))  # fan_out 是输出节点数量
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class AdditiveGuassianNoiseAutoencoder(object):
    """定义一个去噪自编码的类."""

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        """仅用一个隐藏层."""
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        slef.weights = network_weights
