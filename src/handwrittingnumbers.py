"""手写数字识别."""
# !/usr/bin/env python
# coding=utf-8
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 下载并查看样本容量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

# 初始化向量空间
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 实现 Softmax Regression 算法
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义 cross_entropy 交叉熵
y_ = tf.placeholder(tf.float32, [None, 10])  # y_ 表示真实概率分布
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 定义优化算法，这里用 SGD，也即随机梯度下降
train_setup = tf.train.GradientDescentOptimizer(
    0.5).minimize(cross_entropy)  # 下降速率为 0.5， 优化目标设定为 cross_entropy
tf.global_variables_initializer().run()  # 全局参数初始化

# 训练
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 一次选取100条样本
    train_setup.run({x: batch_xs, y_: batch_ys})

# 对准确率进行验证
# tf.argmax 是从一个从 tensor 中寻找最大值的序号
# tf.argmax(y, 1) 就是求各个预测的数字中概率最大的一个
# tf.argmax(y_, 1) 则是寻找样本的真实数字类别
# tf.equal 用来判断预测数字的类别是否为正确类别
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 统计全部样本的预测的 accuracy
# 用 tf.cast 将之前的 correct_prediction 输出的 bool 值转换为float32，再求平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
