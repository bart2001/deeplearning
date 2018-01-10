import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # reproducibility

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

#char_set = list(sorted(set(sentence)))  # 정렬을 해주자. 매번 바뀌니깐 헷갈린다
char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

#print(char_set)
#print(char_dic)
#print(char_idx)

dataX = []
dataY = []

seq_length = 10
for i in range(0, len(sentence) - seq_length):
    x_str = sentence[i : i + seq_length]
    y_str = sentence[i + 1: i + seq_length + 1]
    #print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    dataX.append(x)
    dataY.append(y)

input_dim = len(char_set)   #one hot encoded input data dimension
hidden_size = len(char_set) #ouput dimension
num_classes = len(char_set)
batch_size = len(dataX) #학습 데이터의 갯수

X = tf.placeholder(tf.int32, [None, seq_length])
# [[1, 13, 13, 12, 12, 12, 12, 12], ...] = [[a, b, c, ' ', ',', d, f, e, e, e]]
Y = tf.placeholder(tf.int32, [None, seq_length])
# [[1, 13, 13, 12, 12, 12, 12, 12], ...] = [[a, b, c, ' ', ',', d, f, e, e, e]]
X_one_hot = tf.one_hot(X, num_classes)

cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
# into multi cell
cell = tf.contrib.rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, dtype=tf.float32)
#print(outputs)

# FC(Fully Connected) layer with softmax
X_for_softmax = tf.reshape(outputs, [-1, hidden_size]) #한줄로 만들기
#print(X_for_softmax)
# softmax
softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
softmax_b = tf.get_variable("softmax_b", [num_classes])
outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b
outputs = tf.reshape(outputs, [batch_size, seq_length, num_classes]) #펼치기
#print(outputs)

# 비용함수
weights = tf.ones([batch_size, seq_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(mean_loss)

#학습
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    _, l, results = sess.run([train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})

    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        print('step=', i, 'index=', j, 'str=', ''.join([char_set[t] for t in index]), 'loss=', l)

# 예측을 해보자
predicts = sess.run(outputs, feed_dict={X: dataX})
for j, predict in enumerate(predicts):
    index = np.argmax(predict, axis=1)
    #print([char_set[t] for t in index])
    #print(''.join([char_set[t] for t in index]), end='')

    # 첫번째는 전부 출력
    if j == 0:
        print(''.join([char_set[t] for t in index]), end='')
    # 두번째부터는 끝문자만 출력해서 이어붙이자
    else:
        print(char_set[index[-1]], end='')