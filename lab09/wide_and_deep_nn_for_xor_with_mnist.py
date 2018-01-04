import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 데이터 받아오기
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

#nb_classes = 10 # 0 ~ 9까지의 숫자로 분류

# 입력/출력값
X = tf.placeholder(tf.float32, shape=[None, 28*28]) # 28*28=784
Y = tf.placeholder(tf.float32, shape=[None, 10])

width = 70
#layer1
W1 = tf.Variable(tf.random_normal([784, width]), name='weight1')
b1 = tf.Variable(tf.random_normal([width], name='bias1'))
layer1 = tf.nn.softmax(tf.matmul(X, W1) + b1)


W2 = tf.Variable(tf.random_normal([width, width]), name='weight2')
b2 = tf.Variable(tf.random_normal([width], name='bias2'))
layer2 = tf.nn.softmax(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([width, 10]), name='weight3')
b3 = tf.Variable(tf.random_normal([10], name='bias3'))
hypothesis = tf.nn.softmax(tf.matmul(layer2, W3) + b3)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 모델 테스트
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))

# 정확도 측정 (평균값)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# 학습 횟수
training_epochs = 20
batch_size = 100

# 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 15번 반복
    for epoch in range(training_epochs):
        avg_cost = 0
        # 전체 배치의 횟수(현재는 1 batch에 100개씩 학습하도록 설정)
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            cost_val, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost = avg_cost + (cost_val / total_batch)

        # epoch가 진행될수록 평균비용이 감소하는 것을 볼 수 있다
        print('epoch={:02}, avg_cost={:.2f}'.format(epoch + 1, avg_cost))

    # 정확도 측정 (아래의 두 개의 코드는 같은 출력을 나타냄)
    print("Accuracy=", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
    print("Accuracy=", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
