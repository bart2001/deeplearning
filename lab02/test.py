import tensorflow as tf

session = tf.Session()

a = tf.constant([10, 20])
b = tf.constant([1.0, 2.0])
# 'fetches' can be a singleton
v = session.run(a)
print(v)
# v is the numpy array [10, 20]
# 'fetches' can be a list.
v = session.run([a, b])
print(session.run([a, b]))
# v is a Python list with 2 numpy arrays: the 1-D array [10, 20] and the
# 1-D array [1.0, 2.0]
# 'fetches' can be arbitrary lists, tuples, namedtuple, dicts:
#MyData = collections.namedtuple('MyData', ['a', 'b'])
#v = session.run({'k1': MyData(a, b), 'k2': [b, a]})
# v is a dict with
# v['k1'] is a MyData namedtuple with 'a' (the numpy array [10, 20]) and
# 'b' (the numpy array [1.0, 2.0])
# v['k2'] is