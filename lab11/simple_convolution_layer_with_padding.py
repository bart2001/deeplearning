import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
# 샘플이미지
image = np.array([[[[1],[2],[3]],
                   [[4],[5],[6]],
                   [[7],[8],[9]]]], dtype=np.float32)
# (1, 3, 3, 1) = (n개의 이미지, 3행, 3열, 1개의 컬러(흑백))
print("image.shape=", image.shape)
plt.imshow(image.reshape(3,3), cmap='Greys')
#plt.show()

# image: 1,3,3,1 / fileter: 2,2,1,1 / Stride: 1*1 / Padding: VALID
# image.rank는 filter.rank와 같아야함
# padding = VALID란?
weight = tf.constant(
    [[[[1.]], [[1.]]]
    ,[[[1.]], [[1.]]]]
) # (1, 2, 2, 1)
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
conv2d_img = conv2d.eval()

print("conv2d_img.shape=", conv2d_img.shape)

conv2d_img = np.swapaxes(conv2d_img, 0, 3)

for i, one_img in enumerate(conv2d_img):
        print(one_img.reshape(3,3))
        plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')



