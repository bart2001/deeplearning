import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)\

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# logits = tf.matmul(X, W) + b