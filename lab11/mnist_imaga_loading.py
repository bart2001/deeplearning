import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

img = mnist.train.images[1001].reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.show()

