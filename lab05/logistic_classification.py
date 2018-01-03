import tensorflow as tf

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
# x_data의 shape: (2, 8)
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

# placeholders (나중에 feed_dict를 통해서 넣는다)
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# W = tf.Variable(tf.random_normal([x의 갯수, y의 갯수]), name='weight')
# random한 값을 생성해서 넣는다
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias') #브로드캐스팅에 의해서 아래로 쭉 늘려준다

# 가설
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# 비용함수
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
#cost = tf.reduce_mean(Y * -tf.log(hypothesis) - (1 - Y) * tf.log(1 - hypothesis))

# 학습
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 예측 및 정확도 측정
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) #형변환 -> 0.5보다 클 경우 1
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# 실행
with tf.Session() as sess:
    # 초기화
    sess.run(tf.global_variables_initializer())

    # 학습
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 1000 == 0:
            print('step:', step, 'cost_val:', cost_val)

    # 정확도 측정
    h, c, a, W, b = sess.run([hypothesis, predicted, accuracy, W, b], feed_dict={X: x_data, Y: y_data})
    #print("\nHypothesis:\n", h, "\nCorrect(Y):\n", c, "\nAccuracy:", a, "\nWeight:\n", W, "\nBias:", b)
    print("\nHypothesis:\n", h, "\nPredicted:\n", c, "\nAccuracy:", a, "\nWeight:\n", W, "\nBias:", b)
