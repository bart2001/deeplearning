import tensorflow as tf

# data

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]

# placeholders
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# weight and bias
W = tf.Variable(tf.random_normal([3, 1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis, 행렬의 곱
hypothesis = tf.matmul(X, W) + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)

# train
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initialize
sess.run(tf.global_variables_initializer())

# Run
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train]
        , feed_dict={X: x_data, Y: y_data})

    if step % 100 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction: \n", hy_val)