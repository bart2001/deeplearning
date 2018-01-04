import tensorflow as tf

# 8*4 행렬
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
# 8*3 행렬 (3개의 레이블)
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

X = tf.placeholder("float", [None, 4])  # n*4 행렬
Y = tf.placeholder("float", [None, 3])  # n*3 행렬

W = tf.Variable(tf.random_normal([4, 3]), name='weight')
b = tf.Variable(tf.random_normal([3]), name='bias')

# 가설: 소프트맥스 함수
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# 비용함수(cross-entropy cost funcion)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

# 최소화
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch
with tf.Session() as sess:

    # 초기화
    sess.run(tf.global_variables_initializer())

    # 학습
    for step in range(1001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            cost_val, _, hypothesis_val, one_hot_encoding = sess.run([cost, optimizer, hypothesis, tf.arg_max(hypothesis, 1)], feed_dict={X: x_data, Y: y_data})
            print('step=', step, 'cost_val=', cost_val
                  , "\nhypothesis_val=", hypothesis_val
                  , "\none_hot_encoding=", one_hot_encoding)
            #print('step=', step, 'cost=', sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    print('---------------------------')

    # 테스트 및 one-hot encoding 적용 후의 결과
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.arg_max(a, 1)))

    print('---------------------------')

    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.arg_max(b, 1)))

    print('---------------------------')

    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.arg_max(c, 1)))

    print('---------------------------')

    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})

    print(all, sess.run(tf.arg_max(all, 1)))

    print('--------------')

    d = sess.run(hypothesis, feed_dict={X: [[1, 2, 1, 1]]})
    print(d, sess.run(tf.arg_max(d, 1)))
