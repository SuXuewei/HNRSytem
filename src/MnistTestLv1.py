#!/usr/bin/python
# coding:utf-8

import tensorflow as tf
import input_data
# 加载数据
mnist = input_data.read_data_sets('Mnist_data', one_hot=True)

# x不是一个特定的值，而是一个占位符
# 能够输入任意数量的MNIST图像，每一张图展平成784维的向量
x = tf.placeholder("float", [None, 784])
#  一个Variable代表一个可修改的张量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# y=softmax(Wx+b)
y = tf.nn.softmax(tf.matmul(x, W)+b)
# 添加一个新的占位符用于输入正确值
y_ = tf.placeholder("float", [None, 10])
# 计算交叉熵
# 用tf.reduce_sum 计算张量的所有元素的总和
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# 以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 在Session里面启动模型
sess = tf.Session()
# 初始化变量
init = tf.initialize_all_variables()
sess.run(init)

# 让模型循环训练1000次
for i in range(1000):
    # 随机抓取训练数据中的100个批处理数据点
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 用这些数据点作为参数替换之前的占位符来运行train_step
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 检测预测是否与实际标签匹配,返回一组布尔值
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 把布尔值转换成浮点数，然后取平均值
# [True, False, True, True]变成[1,0,1,1],平均后得0.75
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 计算所学习到的模型在测试数据集上面的正确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

