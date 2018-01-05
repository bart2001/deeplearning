import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 데이터 받아오기
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.set_random_seed(777)  # reproducibility

# 입력/출력값
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([784, 256]))
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([256, 256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

#가설
W3 = tf.Variable(tf.random_normal([256, 10]))
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b3
#logits = tf.matmul(L2, W3) + b3
#hypothesis = tf.nn.softmax(logits)
#94%

#크로스 엔트로피 비용함수로 최소가 되는 비용 계산
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# 모델 테스트
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
# 형변환(True -> 1.0)해서 평균값 구하기
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
            avg_cost += (cost_val / total_batch)

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
    #plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    #plt.show()