import tensorflow as tf
import numpy as np

#csv 읽어오기
#numpy 버전
xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, :-1]
y_data = xy[:, -1:]

X = tf.placeholder(dtype=tf.float32, shape=[None, 8])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal(shape=[8, 1]), name='weight')
b = tf.Variable(tf.random_normal(shape=[1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# 비용함수
cost = tf.reduce_mean((-Y) * tf.log(hypothesis) - (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 결과가 0.5보다 클 경우 1로 변환
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)

# 실제값과 예측값의 차이의 평균으로 정확도 계산
accuracy = tf.reduce_mean(tf.cast(tf.equal(Y, predicted), dtype=tf.float32))
#accuracy = tf.reduce_mean(tf.cast(predicted == Y, dtype=tf.float32)) # 동등비교를 위해서는 반드시 tf.equal() 사용해야 함

with tf.Session() as sess:

    # 텐서보드 기록
    w_hist = tf.summary.histogram("weights", W)
    cost_scalar = tf.summary.scalar("cost", cost)
    accuracy_scalar = tf.summary.scalar("accuracy", accuracy)
    summery = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logdir='./logs')
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _, s, weight_val, bias_val = sess.run(fetches=[cost, train, summery, W, b], feed_dict={X: x_data, Y: y_data})

        writer.add_summary(s, global_step=step)

        if step % 1000 == 0:
            #print('step:', step, 'cost_val:', cost_val, "weight_val:", weight_val, "bias_val:", bias_val)
            print('step:', step, "weight_val:", weight_val, "bias_val:", bias_val)

    h, p, a = sess.run(fetches=[hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    #print("hypothesis:", h, "\npredicted:", p)
    #정확도 출력
    print("accuracy:", a)