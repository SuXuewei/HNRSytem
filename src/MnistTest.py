#!/usr/bin/python
# coding:utf-8

import tensorflow as tf
import input_data

# 加载数据,MNIST数据在当前文件所在同层MNIST_data目录下，输出类型使用的对应索引是1，其余位置是0的输出方式
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#################### 输入层 #########################################################
# 用占位符定义输入图片x与输出类别y_,y_是观测对象的标签数据
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

#################### 隐藏层【学习对象】 #############################################
# 将权重W和偏置b定义为变量,并初始化为0向量，W和b就是索要求解的变量数据
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#################### 输出层 #########################################################
# 类别预测与损失函数，计算的结果存y中
y = tf.nn.softmax(tf.matmul(x, W) + b)

#################### 输出层【学习目标】 #############################################
# 交叉熵损失函数，交叉熵越小，两个数据分布越相似
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#################### 【学习目标】 ###################################################
# 训练模型
# 用梯度下降法让交叉熵下降,步长为0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#################### 计算图构造完毕 #################################################


# 运行交互计算图
sess = tf.InteractiveSession()
# 变量初始化
sess.run(tf.initialize_all_variables())
# 每次加载50个训练样本,然后执行一次train_step,通过feed_dict将x和y_用训练训练数据替代
for i in range(1000):
    # 每次加载50个样本(这个数值的调整可以一定程度上影响到学习效果)
    # 返回一个tuple,元素1为样本,元素2为标签
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# 评估模型
# 用tf.equal来检测预测是与否真实标签匹配
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 将布尔值转换为浮点数来代表对错然后取平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels}))

