import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 데이터 받아오기
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

nb_classes = 10 # 0 ~ 9까지의 숫자로 분류

# 입력/출력값
X = tf.placeholder(tf.float32, [None, 784]) # 28*28=784
Y = tf.placeholder(tf.float32, [None, 10])

#layer1
W1 = tf.Variable(tf.random_normal([784, 784]), name='weight1')
b1 = tf.Variable(tf.random_normal([10], name='bias1'))
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

#layer2
W2 = tf.Variable(tf.random_normal([784, 784]), name='weight2')
b2 = tf.Variable(tf.random_normal([10], name='bias2'))
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

#layer3
W3 = tf.Variable(tf.random_normal([784, 784]), name='weight3')
b3 = tf.Variable(tf.random_normal([10], name='bias3'))
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

#hypothesis
W4 = tf.Variable(tf.random_normal([784, 10]), name='weight4')
b4 = tf.Variable(tf.random_normal([10], name='bias4'))
hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 모델 테스트
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))

# 정확도 측정 (평균값)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# 학습 횟수
training_epochs = 15
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


    # matplot을 활용하여 출력
    import matplotlib.pyplot as plt
    import random

    # 전체 샘플 중에서 임의로 숫자 뽑아오기
    r = random.randint(0, mnist.test.num_examples - 1)
    #
    print("실제=", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("예측=", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

    # 이미지 직접 보기
    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()