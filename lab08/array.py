import tensorflow as tf
import numpy as np
import pprint

#tf.set_random_seed(777)
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

# Simple Array
t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t) #Pretty Print
print(t)
print("dim", t.ndim) #rank
print('shape', t.shape) #shape
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])
print(t[-1])

print('-------------------')

# 2D Array
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
pp.pprint(t)
print(t.ndim)   #rank
print(t.shape)  #shape