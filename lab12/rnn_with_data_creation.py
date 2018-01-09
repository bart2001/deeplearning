import tensorflow as tf
import numpy as np

sample = " if you want you"
idx2char = list(set(sample)) #set으로 중복제거
# ['f', 'o', 'y', 'i', 't', 'w', ' ', 'a', 'u', 'n']
char2idx = {c: i for i, c in enumerate(idx2char)}
# 각 char를 사전처럼 index화 하기 (실행시마다 다름)

# 학습을 위한 샘플데이터
sample_idx = [char2idx[c] for c in sample]

x_data = [sample_idx[:-1]]  # hello: hell (앞부분)
y_data = [sample_idx[1:]]   # hello: ello (뒷부분)

# Hyper parameters
dic_size = len(char2idx)    # rnn input size (one hot size)
rnn_hidden_size = len(char2idx) # rnn output size
num_classes = len(char2idx) # final rnn output size
batch_size = 1  # using one sample data
sequence_length = len(sample) - 1   # ex) size of (hell) or (ello)


X = tf.placeholder(tf.int32, [None, sequence_length]) # X data
Y = tf.placeholder(tf.int32, [None, sequence_length]) # Y label
X_one_hot = tf.one_hot(X, depth=num_classes)  #shape에 주의할것

# rnn cell 만들기
cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})
        result_str = [idx2char[c] for c in np.squeeze(result)]

        if i % 10 == 0:
            print("step={}, loss={:.03f}, Prediction={}".format(i, l, ''.join(result_str)))
