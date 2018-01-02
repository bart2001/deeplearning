import tensorflow as tf

tf.InteractiveSession()

# tf.argmax: 제일 큰 값이 있는 요소의 인덱스를 반환함

x = [[0, 1, 2],
     [2, 1, 0]]

print(tf.argmax(x, axis=0).eval())
print(tf.argmax(x, axis=1).eval())
print(tf.argmax(x, axis=-1).eval())