import tensorflow as tf

# train data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Variable: Tensorflow가 사용하는 변수, Trainable 변수
# 기울기 변수 설정
W = tf.Variable(tf.random_normal([1]), name='weight')
# 절편 변수 설정
b = tf.Variable(tf.random_normal([1]), name='bias')

# 가설: y = x * Weight + Bias
hypothesis = x_train * W + b

# Cost/Loss function 비용함수
# reduce_mean: 평균값을 구한다
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# GradientDescent 경사하강법
# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the grapth in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

# FIt the line
# 학습이 진행될수록 cost는 적어지고 W는 1에 수렴하고 b는 0에 수렴한다
print('step / cost / Weight / Bias')
for step in range(2001):
    sess.run(train)
    # 20번에 한번씩만 출력하기
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))