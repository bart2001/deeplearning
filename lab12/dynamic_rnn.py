import tensorflow as tf
import numpy as np

'''
idx2char = ['h', 'i', 'e', 'l', 'o']
x_data = [[0, 1, 0, 2, 3, 3]]   # hihell
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3
y_data = [[1, 0, 2, 3, 3, 4]]   # ihello
'''
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

#x_data = np.array([[[h, e, l, l, o]]], dtype=np.float32)
x_data = np.array([
    [h, e, l, l, o]
    , [e, o, l, l, l]
    , [l, l, e, e, l]
], dtype=np.float32)

hidden_size = 2
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32
    , sequence_length=[5,3,4])
# 각 batch별로 적용할 sequence의 길이(주로 공백을 처리하기 위한 용도)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(outputs.eval())