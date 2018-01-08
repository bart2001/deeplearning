import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 모델에 대한 정의
class Model:
    # 생성자
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        # 생성과 동시에 호출
        self._build_net()

    # 모델 만들기
    def _build_net(self):
        with tf.variable_scope(self.name):

            self.keep_prob = tf.placeholder(tf.float32)
            self.training = tf.placeholder(tf.bool)

            # input
            self.X = tf.placeholder(tf.float32, [None, 28 * 28])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            # output
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # Convolutional Layer1
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32
                , kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, padding='SAME'
                , strides=2, pool_size=[2, 2])
            dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7
                , training=self.training)

            # Convolutional Layer2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64
                , kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, padding='SAME'
                , strides=2, pool_size=[2, 2])
            droptout2 = tf.layers.dropout(inputs=pool2, rate=0.7
                , training=self.training)

            # Convolutional Layer3
            conv3 = tf.layers.conv2d(inputs=droptout2, filters=128
                , kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, padding='SAME'
                , strides=2, pool_size=[2, 2])
            droptout3 = tf.layers.dropout(inputs=pool3, rate=0.7
                , training=self.training)

            # Fully Connected Layer
            flat = tf.reshape(droptout3, shape=[-1, 128 * 4 * 4])
            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
            droptout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)

            #logits
            self.logits = tf.layers.dense(inputs=droptout4, units=10)

            # 비용함수
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)

            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits
            , feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy
            , feed_dict={self.X: x_test, self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=False):
        return self.sess.run([self.cost, self.optimizer]
            , feed_dict={self.X: x_data, self.Y: y_data, self.training: training})


tf.set_random_seed(777)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.Session()


# 여러 모델을 배열로 저장
models = []
num_models = 7

for m in range(num_models):
    models.append(Model(sess, "model" + str(m)))

sess.run(tf.global_variables_initializer())

training_epochs = 1
batch_size = 100

print("학습시작!!!")
for epoch in range(training_epochs):
    #break
    avg_cost_list = np.zeros(len(models))
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # 각 모델별로 학습시키기
        for m_idx, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch
            print('epoch={}\ncost={}'.format(epoch + 1, avg_cost_list))

print("학습종료!!!")

# 정확도 측정
test_size = len(mnist.test.labels)
predictions = np.zeros(test_size * 10).reshape(test_size, 10)
print("init", predictions)

# 테스트
for m_idx, x in enumerate(models):
    print(m_idx, 'Accuracy=', m.get_accuracy(mnist.test.images, mnist.test.labels))
    p = m.predict(mnist.test.images)
    predictions += p

print("after", predictions)

ensemble_correct_prediction = tf.equal(tf.arg_max(predictions, 1), tf.arg_max(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print("Ensemble accuracy = ", sess.run(ensemble_accuracy))