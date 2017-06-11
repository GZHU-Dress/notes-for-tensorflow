"""AlexNet."""
# coding=utf-8
# !/usr/bin/env python

from datetime import datetime
import os
import math
import time
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
batch_size = 32
num_batches = 100


def print_activations(tensor):
    """定义一个用来显示网络每一层结构的函数."""
    print(tensor.op.name, ' ', tensor.get_shape().as_list())


def inference(images):
    """构建神经网络."""
    parameters = []

    # 第一个卷积层
    with tf.name_scope('conv1') as scope:  # 自动命名
        kernel = tf.Variable(tf.truncated_normal(
            [11, 11, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(
            0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]
    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 /
                     9, beta=0.75, name='lrn1')  # lrn 效果不明显，并且大大降低速度
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[
        1, 2, 2, 1], padding='VALID', name='pool1')  # 取样时不超过边框
    print_activations(pool1)

    # 第二个卷积层
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal(
            [5, 5, 64, 192], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(
            0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        print_activations(conv2)
        parameters += [kernel, biases]
    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 /
                     9, beta=0.75, name='lrn2')  # lrn 效果不明显，并且大大降低速度
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[
        1, 2, 2, 1], padding='VALID', name='pool2')  # 取样时不超过边框
    print_activations(pool2)

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal(
            [3, 3, 192, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(
            0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        print_activations(conv3)
        parameters += [kernel, biases]

    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal(
            [3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(
            0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        print_activations(conv4)
        parameters += [kernel, biases]

    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal(
            [3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(
            0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        print_activations(conv5)
        parameters += [kernel, biases]
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[
                           1, 2, 2, 1], padding='VALID', name='pool5')
    print_activations(pool5)
    return pool5, parameters


def time_tf_run(session, target, info_string):
    """实现一个评估每轮计算时间的函数."""
    num_steps_burn_in = 10  # 预热轮函数，考量从10轮迭代之后的计算时间
    total_duration = 0.0  # 总时间
    total_duration_squared = 0.0  # 平方和
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
        mn = total_duration / num_batches  # 平均耗时
        vr = total_duration_squared / num_batches - mn * mn
        sd = math.sqrt(vr)  # 标准差
        print('%s: %s across %d step, %.3f +/- %.3f sec / batch' %
              (datetime.now(), info_string, num_batches, mn, sd))


def run_benchmark():
    """主函数，这里并不使用 ImageNet 数据集来训练，只使用随机的图片数据测试前馈和反馈计算耗时"""
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,  # 每轮迭代的样本数
                                               image_size,  # 图片尺寸
                                               image_size, 3],  # 色彩通道
                                              dtype=tf.float32,
                                              stddev=1e-1))
        pool5, parameters = inference(images)
        init = tf.global_variables_initializer()  # 初始化所有参数
        sess = tf.Session()
        sess.run(init)

        # 进行 AlexNet 的 forward 计算的评测
        time_tf_run(sess, pool5, "Forward")
        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective, parameters)
        time_tf_run(sess, grad, "Forward-backward")


run_benchmark()
