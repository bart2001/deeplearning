import tensorflow as tf

tf.InteractiveSession()

x = tf.reduce_mean([1, 2], axis=0).eval()
print(x)

print('-----------')

x = [[1., 2.],
     [3., 4.]]
print(tf.reduce_mean(x).eval())
print(tf.reduce_mean(x, axis=None).eval())
print(tf.reduce_mean(x, axis=0).eval())
print(tf.reduce_mean(x, axis=1).eval())
print(tf.reduce_mean(x, axis=-1).eval())

print('----------------------')

t = tf.constant([
    [
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ],
        [
            [13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24]
        ]
    ]
])
print(tf.reduce_mean(t).eval())
print(tf.reduce_mean(t, axis=None).eval())
print(tf.reduce_mean(t, axis=0).eval())
print(tf.reduce_mean(t, axis=1).eval())
print(tf.reduce_mean(t, axis=2).eval())
print(tf.reduce_mean(t, axis=3).eval())
print(tf.reduce_mean(t, axis=-1).eval())


