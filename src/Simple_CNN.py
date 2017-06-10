"""以 LeNet-5 结构实现一个简单的卷积神经网络."""
# coding=utf-8
# !/usr/bin/env python

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


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
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])  # LeNet-5 的第一个池化层

# 第一个卷积层
W_conv1 = weight_variable([5, 5, 1, 32])  # 卷积核尺寸为5x5，一个颜色通道，32个不同的卷积核
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 卷积并加偏置，再做非线性处理
h_pool1 = max_pool_2x2(h_conv1)

# 第二个卷积层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# full connection
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
                                              tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# check error
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training
tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuary.eval(
            feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuary %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("test accuary %g" % accuary.eval(
    feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
