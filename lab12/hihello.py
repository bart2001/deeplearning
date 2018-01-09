import tensorflow as tf
import numpy as np
import pprint

# Session과 PrettyPrinter 선언
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

#hihello라고 말하도록 학습시켜보자!!!

'''
hidden_size = 5 # output from the LSTM
input_dim = 5   # one-hot encoding size 
batch_size = 1  # one sentence (number of sentences)
sequence_length = 6 # ihello -> 6개
'''

# 입력/출력값
idx2char = ['h', 'i', 'e', 'l', 'o']
x_data = [[0, 1, 0, 2, 3, 3]]   # hihell
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3
y_data = [[1, 0, 2, 3, 3, 4]]   # ihello

sequence_length = 6 # 6개의 문자열
hidden_size = 5 # 출력 사이즈
batch_size = 1 # 1개의 데이터셋

X = tf.placeholder(tf.float32, [None, sequence_length, hidden_size]) # X
Y = tf.placeholder(tf.int32, [None, sequence_length]) # Y

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

# 초기화
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=X, initial_state=initial_state, dtype=tf.float32)
weights = tf.ones(shape=[batch_size, sequence_length])

# 비용함수
# logits=outputs는 여기서 임시로 사용하는 방식
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

#예측
prediction = tf.argmax(outputs, axis=2)

#학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):

        # 학습
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})

        # 예측결과값 출력
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(result)

        if i % 100 == 0:
            print("step={}, loss={:.02f}, prediction={}, true Y={}".format(i, l, result, y_data))

    # print char using dic
    result_str = [idx2char[c] for c in np.squeeze(result)]
    print("result=", result)
    print("np.squeeze(result)=", np.squeeze(result))
    print("result_str=", result_str)
    print("Prediction str:", ''.join(result_str))