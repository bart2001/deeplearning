import tensorflow as tf

a = tf.random_normal([3, 4,3])


print(tf.Session().run(a))