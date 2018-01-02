import tensorflow as tf

tf.InteractiveSession()

t = tf.one_hot([[0], [1], [2], [0]], depth=3)
print(t.eval())

# one-hot encoding을 할 경우 rank가 늘어나기 때문에 rank를 보통 줄여준다
print(tf.reshape(t, shape=[-1, 3]).eval())



