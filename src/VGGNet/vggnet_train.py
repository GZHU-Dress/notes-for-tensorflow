"""探索卷积神经网络的深度与其性能之间的关系.这里采用 VGGNet-16."""
# coding=utf-8
# !/usr/bin/env python


import time
import math
import tensorflow as tf
from datetime import datetime


def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    """
    用于创建卷积层并把本层的参数存入参数列表.

    参数:
        input_op 输入的 tensor
        name 这一层的名字
        kh kernel height 卷积层核高
        kw kernel weight 卷积层核宽
        n_out 卷积核数量即输出通道数
        dh 步长的高
        dw 步长的宽
        p 参数列表
    """
    n_in = input_op.get_shape()[-1].value
    # 获得 input_op 的通道数，比如图片尺寸为 224x224x3，那么通道数就是3

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(
            scope + 'w',
            shape=[kh, kw, n_in, n_out], dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1),
                            padding='SAME')
        bias_init_val = tf.constant(
            0.0, shape=[n_out], dtype=tf.float32)  # 赋值为0
        biases = tf.Variable(bias_init_val, trainable=True,
                             name='b')  # 转成可训练的参数
        z = tf.nn.bias_add(conv, biases)  # 将卷积结果 conv 与 bias 相加
        activation = tf.nn.relu(z, name=scope)  # 使用 ReLU 做非线性处理
        p += [kernel, biases]  # 将创建时用到的 kernel 和 biases 添加进参数列表 p
        return activation


def fc_op(input_op, name, n_out, p):
    """定义全连接层的创建函数."""
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(
            scope + 'w', shape=[n_in, n_out], dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.Variable(tf.constant(
            0.1, shape=[n_out], dtype=tf.float32), name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name='scope')
        p += [kernel, biases]
        return activation


def mpool_op(input_op, name, kh, kw, dh, dw):
    """定义最大池化层."""
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)


def inference_op(input_op, keep_prob):
    """第一部分，由两个卷积层和一个最大池化层组成."""
    p = []

    # 第一段
    conv1_1 = conv_op(input_op, name='conv1_1', kh=3, kw=3,
                      n_out=64, dh=1, dw=1, p=p)
    conv1_2 = conv_op(conv1_1, name='conv1_2', kh=3, kw=3,
                      n_out=64, dh=1, dw=1, p=p)
    pool1 = mpool_op(conv1_2, name='pool1', kh=2, kw=2, dw=2, dh=2)

    # 第二段
    conv2_1 = conv_op(pool1, name='conv2_1', kh=3, kw=3,
                      n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv_op(conv2_1, name='conv2_2', kh=3, kw=3,
                      n_out=128, dh=1, dw=1, p=p)
    pool2 = mpool_op(conv2_2, name='pool2', kh=2, kw=2, dw=2, dh=2)

    # 第三段
    conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3,
                      n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3,
                      n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3,
                      n_out=256, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_3, name='pool3', kh=2, kw=2, dw=2, dh=2)

    # 第四段
    conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3,
                      n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3,
                      n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3,
                      n_out=512, dh=1, dw=1, p=p)
    pool4 = mpool_op(conv4_3, name='pool4', kh=2, kw=2, dw=2, dh=2)

    # 第五段
    conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3,
                      n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv4_1, name='conv5_2', kh=3, kw=3,
                      n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3,
                      n_out=512, dh=1, dw=1, p=p)
    pool5 = mpool_op(conv5_3, name='pool5', kh=2, kw=2, dw=2, dh=2)

    # 扁平化，将每个样本化为7x7x512=25088的一维向量
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name='resh1')

    # 连接一个隐含节点数为 4096 的全连接层，激活函数为 ReLU。
    # 然后连接一个 Dropout 层， 在训练时节点保留率为0.5，预测时为1
    fc6 = fc_op(resh1, name='fc6', n_out=4096, p=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name='fc6_drop')

    # 第二个全连接层
    fc7 = fc_op(fc6_drop, name='fc7', n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name='fc7_drop')

    # 第三个全连接层
    fc8 = fc_op(fc7_drop, name='fc8', n_out=1000, p=p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc8, p
