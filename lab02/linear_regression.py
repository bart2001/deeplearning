import tensorflow as tf

# 학습 데이터
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Variable: Tensorflow가 사용하는 변수, Trainable Variable (학습을 통해서 구해야하는 값)
# 기울기/절편 변수 설정
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 가설: y = x * Weight + Bias
hypothesis = x_train * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# GradientDescent 경사하강법
# Minimize
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#train = optimizer.minimize(cost)
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 세션생성
sess = tf.Session()

# 초기화
sess.run(tf.global_variables_initializer())

# FIt the line
# 학습이 진행될수록 cost는 적어지고 W는 1에 수렴하고 b는 0에 수렴한다
for step in range(2001):

    # 학습
    sess.run(train)

    # 20번에 한번씩만 출력하기
    if step % 20 == 0:
        print('step=', step, 'cost=', sess.run(cost), 'weight=', sess.run(W), 'bias=', sess.run(b))

# 학습이 진행될수록 cost는 적어지고 weight=1, bias=0에 가까워진다