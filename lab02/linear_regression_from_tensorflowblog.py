# 출처: https://tensorflow.blog/2-%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C%EC%9A%B0-%EC%84%A0%ED%98%95-%ED%9A%8C%EA%B7%80%EB%B6%84%EC%84%9D-first-contact-with-tensorflow/

# 1. 데이터 생성
import numpy as np

num_points = 1000
vectors_set = []

for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

print(x_data)
print(y_data)

# 2. 데이터출력
import matplotlib.pyplot as plt

plt.plot(x_data, y_data, 'ro', label='Original data')
plt.legend()
#plt.show()

# 3.회귀분석
import tensorflow as tf

W = tf.Variable(tf.random_normal([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# 비용함수
#loss = tf.reduce_mean(tf.square(y - y_data))
#optimizer = tf.train.GradientDescentOptimizer(0.5)
#train = optimizer.minimize(loss)
train = tf.train.GradientDescentOptimizer(0.5).minimize(tf.reduce_mean(tf.reduce_mean(tf.square(y - y_data))))

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#train = optimizer.minimize(cost)
#train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 초기화
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(8):

    sess.run(train)
    print(step, sess.run(W), sess.run(b))

    '''
    #Graphic display    
    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
    plt.xlabel('x')
    plt.xlim(-2, 2)
    plt.ylim(0.1, 0.6)
    plt.ylabel('y')
    plt.legend()
    plt.show()
    '''

