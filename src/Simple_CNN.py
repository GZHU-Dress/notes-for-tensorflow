"""以 LeNet-5 结构实现一个简单的卷积神经网络."""
# coding=utf-8
# !/usr/bin/env python

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])  # LeNet-5 的第一个池化层

# 第一个卷积层
W_conv1 = weight_variable([5, 5, 1, 32])  # 卷积核尺寸为5x5，一个颜色通道，32个不同的卷积核
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 卷积并加偏置，再做非线性处理
h_pool1 = max_pool_2x2(h_conv1)

# 第二个卷积层
W_Conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_Conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])


def weight_variable(shape):
    """给权重制造一些随机噪声打破完全对称."""
    initial = tf.truncated_normal(shape, stddev=1.0)
    return tf.Variable(initial)


def bias_variable(shape):
    """添加一些小偏差来避免死亡节点."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """卷积层，采用二维卷积函数."""
    # x 是输入
    # W 是卷积的参数，比如 [5, 5, 1, 32] 前面两个数字代表卷积核尺寸，第三个数字代表有多少个
    #    channel，如果是灰度图像则为1，最后一个数字代表卷积核的数量
    # strides 代表卷积模板移动步长，1代表每个点都会处理
    # padding 代表边界处理方式，SAME 则表示输出与输入保持同样的尺寸
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """最大池化函数"""
    # 使用2x2的最大池化，将一个2x2的像素块降为1x1的像素。
    # 最大池化会保留原始像素块中灰度值最高的哪一个像素，即保留最显著特征.
    # 因为希望整体上缩小图片尺寸，因此strides也设置为2x2，即横竖方向以2为步长
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]), padding = 'SAME')
