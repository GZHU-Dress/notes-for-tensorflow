"""带有隐藏层和 ReLU 激活函数的手写数字识别."""
# coding=utf-8
# !/usr/bin/env python


import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()  # 创建一个默认的 session
in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal(
    [in_units, h1_units], stddev=0.1))  # 标准差0.1
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)  # ReLU
# 随机丢弃一些节点数据，在训练的时候 keep_prob < 1 用以制造随机性，防止过拟合
# 在预测的时候 keep_prob = 1
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
                                              tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)  # 交叉信息熵
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images,
                     y_: mnist.test.labels, keep_prob: 1.0}))  # 准确度约为98%
