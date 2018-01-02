import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # = tf.add(a,b)

sess = tf.Session()

## 값에 대한 동적할당(feeding)
print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1,3], b: [2, 4]}))

print('-------------------------')

input_data = [1,2,3,4,5]
x = tf.placeholder(dtype=tf.float32)
y = x * 2
sess = tf.Session()
result = sess.run(y,feed_dict={x:input_data})
print(result)