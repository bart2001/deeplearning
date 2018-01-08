import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
# 샘플이미지
image = np.array([[[[4],[3]],
                   [[2],[1]]]], dtype=np.float32)
# ksize: pooling을 해서 뽑아낼 tensor의 shape
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1]
    , strides=[1, 1, 1, 1], padding='SAME')
print(pool.shape)
print(pool.eval())

'''
[[[[ 4.]
   [ 3.]]

  [[ 2.]
   [ 1.]]]]
'''



