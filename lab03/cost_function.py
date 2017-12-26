import tensorflow as tf
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)
# Y = X * W 모델
hypothesis = X * W

# 비용함수
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Launch
sess = tf.Session()

# Initialize
sess.run(tf.global_variables_initializer())

# Variables for plotting cost function
W_val = []
cost_val = []

# 학습
for i in range (-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

# Show the cost funcion
plt.plot(W_val, cost_val)
plt.show()