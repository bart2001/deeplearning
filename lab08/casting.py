import tensorflow as tf

tf.InteractiveSession()

# casting: 원하는 형태로 형을 변환해준다

print(tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval())

print(tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval())

