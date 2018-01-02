import tensorflow as tf

tf.InteractiveSession()

x = [[1, 2], [3, 4]]

print(tf.reduce_sum(x).eval())
print(tf.reduce_sum(x, axis=0).eval())
print(tf.reduce_sum(x, axis=1).eval())
print(tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval())