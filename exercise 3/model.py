import tensorflow as tf
import numpy as np


class Model:

    def __init__(self, filter_size, num_filters=32, learning_rate=0.0001, history_length=1):
        self.learning_rate = learning_rate

        self.x_image = tf.placeholder(tf.float32, shape=[None, 96, 96, None], name='x')
        self.y_ = tf.placeholder(tf.float32, shape=[None, 5], name='y_')
        # self.y_conv = tf.placeholder(tf.float32, shape=[None, 5], name='y_conv')
        batch_size = tf.shape(self.x_image)[0]

        self.W_conv1 = tf.get_variable("wc1", [filter_size*2, 2*filter_size, history_length, num_filters], initializer=tf.contrib.layers.xavier_initializer()) #should be bigger since image is 96x96
        self.b_conv1 = self.bias_variable([num_filters])

        h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)

        self.W_conv2 = tf.get_variable("wc2", [filter_size, filter_size, num_filters, num_filters], initializer=tf.contrib.layers.xavier_initializer())
        self.b_conv2 = self.bias_variable([num_filters])

        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, self.W_conv2) + self.b_conv2)

        self.W_conv3 = tf.get_variable("wc3", [filter_size, filter_size, num_filters, num_filters], initializer=tf.contrib.layers.xavier_initializer())
        self.b_conv3 = self.bias_variable([num_filters])

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, self.W_conv3) + self.b_conv3)

        self.W_conv4 = tf.get_variable("wc4", [filter_size, filter_size, num_filters, num_filters], initializer=tf.contrib.layers.xavier_initializer()) 
        self.b_conv4 = self.bias_variable([num_filters])

        h_conv4 = tf.nn.relu(self.conv2d(h_conv3, self.W_conv4) + self.b_conv4)

        h_conv4_flat = tf.contrib.layers.flatten(h_conv4)
        h_fc1 = tf.layers.dense(h_conv4_flat, 400, activation=tf.nn.relu)
        h_fc1_drop = tf.layers.dropout(h_fc1, 0.8)

        h_fc2 = tf.layers.dense(h_fc1_drop, 400, activation=tf.nn.relu)
        h_fc2_drop = tf.layers.dropout(h_fc2, 0.8)

        h_fc3 = tf.layers.dense(h_fc2_drop, 50, activation=tf.nn.relu)

        lstm_layer = tf.contrib.rnn.LSTMCell(num_units=128)
        lstm_layer = tf.contrib.rnn.DropoutWrapper(lstm_layer, output_keep_prob=0.8)
        lstm_layer = tf.contrib.rnn.MultiRNNCell(cells=[lstm_layer])

        initial_state = lstm_layer.zero_state(batch_size=batch_size, dtype=tf.float32)
        input = tf.expand_dims(h_fc3, axis=1)

        output, final_state = tf.nn.dynamic_rnn(cell=lstm_layer, inputs=input, dtype=tf.float32,
                                                initial_state=initial_state)
        output = tf.reshape(output, [-1, 128])

        self.logits = tf.layers.dense(output, 5, activation=tf.nn.relu)

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_)
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy)
        # self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))

        self.correct_prediction = tf.equal(tf.argmax(self.logits,1), tf.argmax(self.y_,1))    
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name='accuracy')

        self.prediction = tf.argmax(self.logits, axis=1)

        self.sess = tf.Session()

        self.saver = tf.train.Saver()

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)

    def save(self, file_name):
        self.saver.save(self.sess, file_name)
        return file_name

    def bias_variable(self, shape):
        initial = tf.zeros(shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')
