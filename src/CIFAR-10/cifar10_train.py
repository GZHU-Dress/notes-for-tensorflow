"""
CIFAR-10 一个包含60000张32x32的彩色图像，一共分为10类:airplane, automobile, bird,
cat, deer, dog, frog, horse, ship, truck.训练集50000张，测试集10000张.
"""
# coding=utf-8
# !/usr/bin/env python

import os
import time
import cifar10
import cifar10_input
import numpy as np
import tensorflow as tf


def variable_with_weight_loss(shape, stddev, wl):
    """初始化 weight 的函数."""
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


max_steps = 3000
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'
cifar10.maybe_download_and_extract()
images_train, labels_train = cifar10_input.distorted_inputs(
    data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(
    eval_data=True, data_dir=data_dir, batch_size=batch_size)
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.float32, [batch_size])

weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], strides=[
                       1, 2, 2, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[
                       1, 2, 2, 1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], sttdev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
