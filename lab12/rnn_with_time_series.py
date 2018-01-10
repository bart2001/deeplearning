import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# many to one (과거 6일의 데이터를 기반으로 7일째의 데이터를 예측)

timesteps = seq_length = 7
input_dim = 5
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
    print('X=', _x, "-> Y=", _y)

    dataX.append(_x)
    dataX.append(_y)

train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX = np.array(dataX[0:train_size])
testX = np.array(dataX[train_size:len(dataX)])
trainY = np.array(dataY[0:train_size])
trainY = np.array(dataY[train_size:len(dataX)])

# placeholders
X = tf.placeholder(tf.float32, [None, seq_length, input_dim])
Y = tf.placeholder(tf.float32, [None, 1])

