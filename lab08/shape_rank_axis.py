import tensorflow as tf

tf.InteractiveSession()

#shape
t = tf.constant([1, 2, 3, 4])
print("shape:", tf.shape(t).eval())

t = tf.constant([
    [1, 2], [3, 4]
])
print("shape:", tf.shape(t).eval())

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
print("shape:", tf.shape(t).eval())

#axis: rank에 대한 번호 ex) -1은 제일 안쪽에 있는 값

