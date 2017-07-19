import tensorflow as tf
import numpy as np


def read_data(file_name, delimiter=','):
    return np.loadtxt(file_name, delimiter=delimiter)
# 将4.1节原始数据，转换为4.2节中数据


def init_data(input_data):
    input_x = input_data[:, 0:-1]
    input_y = input_data[:, -1].reshape(m, 1)
    input_x = np.column_stack(
        (np.ones(input_data.shape[0]), input_x)).reshape(m, n)
    return input_x, input_y
# 对应3.5节中，均值标准化


def feature_normalization(input_x):
    for i in range(1, input_x.shape[1]):
        mn = np.mean(input_x[:, i], dtype=np.float64)
        std = np.std(input_x[:, i], ddof=1, dtype=np.float64)
        input_x[:, i] = (input_x[:, i] - mn) / std
    return input_x
# 实现线性回归模型


def tensor_flow_run(input_x, input_y):
    # 初始化placeholder，后续将特征矩阵与结果向量导入
    X = tf.placeholder("float64")
    Y = tf.placeholder("float64")
    # 初始化theta
    theta = tf.Variable(tf.zeros([n, 1], dtype=tf.float64))
    # 构建cost函数，即2.2节公式
    cost = tf.reduce_sum(tf.square(tf.matmul(X, theta) - Y)) / 2 / m
    # 初始化梯度下降算法的学习率与策略
    train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)
    with tf.Session() as sess:
        # 初始化所有参数
        init = tf.global_variables_initializer()
        sess.run(init)
        # 进行梯度下降
        for i in range(ITERATIONS):
            # 导入特征矩阵与结果向量，进行一次梯度下降计算
            # 调用TensorFlow库函数，实现4.3节内容
            sess.run(train_op, feed_dict={X: input_x, Y: input_y})
        # 输出最终theta参数
        print(sess.run(theta))


# 初始化学习率与迭代次数
LEARNING_RATE = 0.01
ITERATIONS = 1000
# 输入数据
data = read_data('data1.txt', ',')
# 计算参数数量以及样本数量
m = data.shape[0]
n = data.shape[1]
# 对输入数据进行处理
datas = init_data(data)
x = datas[0]
y = datas[1]
# 特征标准化（均值标准化）
x = feature_normalization(x)
# 使用TensorFlow实现线性回归模型的梯度下降
tensor_flow_run(x, y)
