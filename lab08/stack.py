import tensorflow as tf

tf.InteractiveSession()

# stack 쌓는다

x = [1, 4]
y = [2, 5]
z = [3, 6]

print(tf.stack([x, y, z]).eval())
print(tf.stack([x, y, z], axis=1).eval())
print(tf.stack([x, y, z], axis=0).eval())