import tensorflow as tf
import numpy  as np

import matplotlib.pyplot as plt
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3


# plt.plot(train_X, train_Y)
# plt.show()



## ------------------ 搭建模型 ---------------------------------------------
# 使用占位符定义输入节点
X = tf.placeholder("float")
Y = tf.placeholder("float")

# 定义变量
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')


# 定义模型
z = tf.multiply(W, X) + b

# 优化
cost = tf.reduce_mean(tf.square(Y - z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


## ------------------ 训练模型 ---------------------------------------------


init = tf.global_variables_initializer()
training_epochs = 20
display_step = 2


with tf.Session() as sess:
    sess.run(init)
    plot_data = {
        'step':[],   
        'loss':[],
        }
    for epoch in range(training_epochs):
        for x,y in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y:y})
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
            print (loss)


