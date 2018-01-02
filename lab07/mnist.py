import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

X = tf.placeholder(tf.float32, [None, 28 * 28])
Y = tf.placeholder(tf.float32, [None, nb_classes])

# epoch
# batch
# iteration

print(mnist)
