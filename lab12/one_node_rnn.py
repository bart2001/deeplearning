import tensorflow as tf
import numpy as np
import pprint

# Session과 PrettyPrinter 선언
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

# One cell RNN: input_dim=4 -> output_dimension=2
# because hidden_size is 4
hidden_size = 2
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)

# shape(1,1,4) -> shape(1,1,2)
x_data = np.array([[[1, 0, 0, 0]]], dtype=np.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

sess.run(tf.global_variables_initializer())

pp.pprint(x_data)
pp.pprint(outputs.eval())
