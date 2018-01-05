import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sees = tf.InteractiveSession()
image = np.array([[[[1],[2],[3]],
                   [[4],[5],[6]],
                   [[7],[8],[9]]]], dtype=np.float32)

print("image.shape=", image.shape)
#plt.imshow(image.reshape(3,3), cmap='Greys')
#plt.show()

#image: 1,3,3,1 / fileter: 2,2,1,1 / Stride: 1*1 / Padding: VALID
weight = tf.constant(
    [
        [
            [[1.]], [[1.]]],
            [[[1.]], [[1.]]
        ]
    ]
)

conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='VALID')
conv2d_img = conv2d.eval()

print("conv2d_img.shape=", conv2d_img.shape)

