"""
CIFAR-10 一个包含60000张32x32的彩色图像，一共分为10类:airplane, automobile, bird,
cat, deer, dog, frog, horse, ship, truck.训练集50000张，测试集10000张.
"""
# coding=utf-8
# !/usr/bin/env python

import os
import math
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


def loss(logits, labels):
    """计算 CNN 的 loss."""
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name="cross_entropy_per_example")
    # 把 softmax 的计算和 cross entropy loss 的计算合并在了一起
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 因为服务器是32核，64线程的，如果不够可以到 cifar10_input.py 文件中修改
max_steps = 3000  # 训练轮数
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'  # 指定默认下载路径
cifar10.maybe_download_and_extract()
images_train, labels_train = cifar10_input.distorted_inputs(
    data_dir=data_dir, batch_size=batch_size)  # 进行数据增强
images_test, labels_test = cifar10_input.inputs(
    eval_data=True, data_dir=data_dir, batch_size=batch_size)
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])  # RGB
label_holder = tf.placeholder(tf.int32, [batch_size])

# 创建第一个卷积层，卷积核5x5，3个颜色通道，64个卷积核，设置 weight 初始化函数的标准差为0.05
# 因为不对第一个卷积层做 L2 正则，所以 wl=0.
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[
                       1, 2, 2, 1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)  # 侧抑制

# 创建第二个卷积层
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))  # 初始化为0.1
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[
                       1, 2, 2, 1], padding='SAME')

# 创建一个全连接层
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 创建第二个全连接层，隐含节点数下降一半
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# 最后一层，softmax 放在了计算 loss 的部分
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)

# 计算 CNN 的 loss
loss = loss(logits, label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)  # 获得最终 loss
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)  # 输出分数最高的那一类的准确率
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()

for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time
    if step % 10 == 0:
        # 每10个 step 会计算并展示当前的 loss
        # 每秒钟能训练的样本数量，以及训练一个 batch 所需要的时间
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        format_str = (
            'step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

num_examples = 10000
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={
        image_holder: image_batch, label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1

precision = true_count / total_sample_count  # 计算最后准确率
print('precision @ 1 = %.3f' % precision)
