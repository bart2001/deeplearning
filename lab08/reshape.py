import numpy as np
import tensorflow as tf

tf.InteractiveSession()

t = np.array([
    [
        [0, 1, 2],
        [3, 4, 5]
    ],
    [
        [6, 7, 8],
        [9, 10, 11]
    ]
])
print(t.shape)

print(tf.reshape(t, shape=[-1, 1]).eval())

print(tf.reshape(t, shape=[-1, 1, 3]).eval())