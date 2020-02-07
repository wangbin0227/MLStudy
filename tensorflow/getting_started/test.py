import tensorflow as tf

a = tf.Variable([[1,10,100], [1,2,3]], dtype=tf.float32)
b = tf.Variable([[1,2,3], [4,5,6]], dtype=tf.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
c = tf.log(a)
print (sess.run(c))
print (sess.run(a))

print (sess.run(a * b))

print (sess.run(tf.reduce_sum(a * b, reduction_indices=1)))
