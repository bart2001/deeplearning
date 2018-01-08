import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# input
X = tf.placeholder(tf.float32, [None, 28*28])
# cnn을 하기 위한 reshape (n개, 28행, 28열, 흑/백)
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

# Layer1 = Weight1 = (3행, 3열, 흑/백, 32개의 필터)
# img = (?, 28, 28, 32)
# conv = (?, 28, 28, 32)
# pool = (?, 14, 14 32)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

# Layer2
# img = (?, 14, 14, 32)
# conv = (?, 14, 14, 64)
# pool = (?, 7, 7, 64) (, , , number of filters)
# reshape = (?, 3136)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.reshape(L2, [-1, 7 * 7 * 64]) # 펼치기

# 가설
W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b

# 비용함수
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
print("학습시작")
training_epochs = 15
batch_size = 100
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    print("number of examples=", mnist.train.num_examples, "total_batch=", total_batch)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
        avg_cost += c / total_batch
        if i % 10 == 0:
            print("epoch=", epoch, "batch=", i, "avg_cost=", avg_cost)

    print('Epoch=', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

print("학습종료")

# 모델 테스트
correct_prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Prediction: ", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))