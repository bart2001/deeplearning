import tensorflow as tf

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 가설
hypothesis = X * W + b

# 비용함수
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch
sess = tf.Session()

# Initialize
sess.run(tf.global_variables_initializer())

# 2001번의 시행
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train]
        , feed_dict={X: [1,2,3,4,5], Y:[2.1,3.1,4.1,5.1,6.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# 테스트
print(sess.run(hypothesis, feed_dict={X:[5]}))
print(sess.run(hypothesis, feed_dict={X:[2.5]}))
print(sess.run(hypothesis, feed_dict={X:[1.5, 3.5]}))