import tensorflow as tf

sess = tf.InteractiveSession()

matrix1 = tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[1.], [2.]])
matrix3 = tf.matmul(matrix1, matrix2)
print(matrix1.eval())
#print(matrix3)

# 직접할 ㄱ셩우 다른 결과가 나옴
#print((matrix1 * matrix2).eval())