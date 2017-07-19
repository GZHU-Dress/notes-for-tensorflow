"""A linear regression way by tensorflow."""
# coding=utf-8
# !/usr/bin/env python


import tensorflow as tf
import numpy as np


def read_data(file_name,
              delimiter=','):
    """Read data."""
    return np.loadtxt(file_name, delimiter=delimiter)


def init_data(input_data):
    input_x = input_data[:, 0:-1]
    input_y = input_data[:, -1].reshape(m, 1)
    input_x = np.column_stack(
        (np.ones(input_data.shape[0]), input_x)).reshape(m, n)
    return input_x, input_y


def feature_normalization(input_x):
    for i in range(1, input_x.shape[1]):
        mu = np.mean(input_x[:, i], dtype=np.float64)
        std = np.std(input_x[:, i], ddof=1, dtype=np.float64)
        input_x[:, i] = (input_x[:, i] - mu) / std
    return input_x


def run(input_x, input_y):
    X = tf.placeholder('float64')
    Y = tf.placeholder('float64')
    theta = tf.Variable(tf.zeros([n, 1], dtype=tf.float64))
    cost = tf.reduce_sum(tf.square(tf.matmul(X, theta) - Y)) / 2 / m
    train_op = tf.train.GradientDescentOptimizer(
        learning_rate=rate).minimize(cost)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(times):
            print(i)
            sess.run(train_op, feed_dict={X: input_x, Y: input_y})
        print(sess.run(theta))


rate = 0.01
times = 1000
data = read_data('data1.txt')
m = data.shape[0]
n = data.shape[1]
datas = init_data(data)
x = datas[0]
y = datas[1]
x = feature_normalization(x)
run(x, y)
