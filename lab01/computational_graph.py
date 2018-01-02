import tensorflow as tf

# 상수
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

# 이렇게 출력하면 결과값이 아닌 그래프의 요소로 표현된다
print("node1:", node1, "node2:", node2)
print("node3:", node3)

sess = tf.Session()
print("sess.run(node1, node2): ", sess.run([node1, node2]))
print("sess.run(node3): ", sess.run(node3))



