import tensorflow as tf

tf.InteractiveSession()

print(tf.squeeze([[0], [1], [2]]).eval())

print(tf.expand_dims([0, 1, 2], 1).eval())