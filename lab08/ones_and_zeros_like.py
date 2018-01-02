import tensorflow as tf

tf.InteractiveSession()

x = [[0, 1, 2], [2, 1, 0]]

# 0 혹은 1로 구성된 tensor를 반환한다

print(tf.ones_like(x).eval())
print(tf.zeros_like(x).eval())