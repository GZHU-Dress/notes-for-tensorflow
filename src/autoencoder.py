"""TF 实现自编码器."""
# !/usr/bin/env python
# coding=utf-8


import os
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class AdditiveGuassianNoiseAutoencoder(object):
    """定义一个去噪自编码的类."""

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        """仅用一个隐藏层."""
        self.n_input = n_input  # 输入变量数
        self.n_hidden = n_hidden  # 隐藏层节点数
        self.transfer = transfer_function  # 隐藏层激活函数，默认 softplus
        self.scale = tf.placeholder(tf.float32)  # 高斯噪声系数，默认0.1
        # optimizer 优化器，默认为 Adam
        self.training_scale = scale
        network_weights = self._initialize_weights()  # 参数初始化
        self.weights = network_weights
        # 定义网络结构
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(
            self.x + scale * tf.random_normal((n_input,)), self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(
            tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        # 定义损失函数(代价函数)，这里直接使用平方误差(Squared Error)
        self.cost = 0.5 * \
            tf.reduce_sum(
                tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        # tf.pow 矩阵次幂
        self.optimizer = optimizer.minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        """初始化参数."""
        all_weights = dict()
        all_weights['w1'] = tf.Variable(
            xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(
            tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(
            tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(
            tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        """训练数据并返回 cost."""
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={
                                  self.x: X, self.scale: self.training_scale})
        # feed_dict 用于训练的数据
        return cost

    def calc_total_cost(self, X):
        """只求损失 cost 的函数，不会触发训练操作."""
        cost_total = self.sess.run(self.cost, feed_dict={
                                   self.x: X, self.scale: self.training_scale})
        return cost_total

    def transform(self, X):
        """返回自编码器隐藏层的输出结果."""
        result = self.sess.run(self.hidden, feed_dict={
                               self.x: X, self.scale: self.training_scale})
        return result

    def generate(self, hidden=None):
        """将隐藏层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据."""
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        """整体运行一遍复原过程，包括提取高阶特征和通过高阶特征复原数据."""
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    def get_Weights(self):
        """获取隐藏层权重 w1."""
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        """获取隐藏层偏置系数 b1."""
        return self.sess.run(self.weights['b1'])


def standard_scale(X_train, X_test):
    """标准化数据，也即标准差为1的分布."""
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def random_block(data, batch_size):
    """从数据中获取一个随机的block，做一个不放回抽样，得到batch_size."""
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


def xavier_init(fan_in, fan_out, constant=1):
    """Xaiver 初始化器，自动优化权重."""
    low = -constant * np.sqrt(6 / (fan_in + fan_out))  # fan_in 是输入节点数量
    high = constant * np.sqrt(6 / (fan_in + fan_out))  # fan_out 是输出节点数量
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # 读取数据
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
n_samples = int(mnist.train.num_examples)
training_epochs = 20  # 最大训练轮数为20
batch_size = 128  # batch_size 设置为128
display_step = 1  # 每隔一轮显示一次损失 cost
autoencoder = AdditiveGuassianNoiseAutoencoder(
    n_input=784,
    n_hidden=200,
    transfer_function=tf.nn.softplus,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
    scale=0.01)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = random_block(X_train, batch_size)
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size
    if epoch % display_step == 0:
        print("Epoch: ", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
