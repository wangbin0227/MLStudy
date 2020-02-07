import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

### 下载数据
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.reset_default_graph()

### 定义占位符
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


#test = tf.Variable(tf.random_normal([4, 784]))
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.ones([10]))
### 问题：张量的加法，softmax等函数定义
pred = tf.nn.softmax(tf.matmul(x, W) + b)

### 定义损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))


### 定义优化器
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


### 训练模型
training_epochs = 25
batch_size = 100
display_step = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y:batch_ys})
            avg_cost += c / total_batch
        if (epoch + 1) / display_step == 0:
            print (epoch + 1, ": .9f".format(avg_cost))
    print ("Finished")


print (sess.run(res))
print (sess.run(res2))

print (W)
print (W.shape)
print (b)
print (b.shape)
