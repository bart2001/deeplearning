import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# many to one (과거 6일의 데이터를 기반으로 7일째의 데이터를 예측)

timesteps = seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1  #hidden_size

xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1] #시간순으로 revearse
#print(xy[0])
#print('------------')
xy = MinMaxScaler().fit_transform(xy)   # 데이터 정규화
#print(xy[0])
#exit(0)
x = xy  # input=1st~5th
y = xy[:, [-1]] # output=5th
#print(x)
#exit()
#print(y.shape)

dataX = []
dataY = []

for i in range(0, len(y) - seq_length):
    _x = x[i : i + seq_length]  #1~7일치의 개시/최고/최저/사이즈/종료 값
    _y = y[i + seq_length]  #8일차의 종료값
    #print('X=', _x, "-> Y=", _y)

    dataX.append(_x)
    dataY.append(_y)


train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX = np.array(dataX[0:train_size])
testX = np.array(dataX[train_size:len(dataX)])
trainY = np.array(dataY[0:train_size])
testY = np.array(dataY[train_size:len(dataY)])

# placeholders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

#LSTM Cell
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
outputs, _state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)

#cost function
loss = tf.reduce_sum(tf.square(Y_pred - Y))
optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(loss)

#학습
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    _, l = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
    if i % 100 == 0:
        print("step={}, loss={}".format(i, l))

testPredict = sess.run(Y_pred, feed_dict={X: testX})
plt.plot(testY)
plt.plot(testPredict)
plt.show()