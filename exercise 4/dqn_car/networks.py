import tensorflow as tf
import numpy as np

# TODO: add your Convolutional Neural Network for the CarRacing environment.

class NeuralNetwork():
    """
    Neural Network class based on TensorFlow.
    """
    def __init__(self, state_dim, num_actions, num_filters=32, filter_size=5, hidden=20, lr=1e-4):
        self._build_model(state_dim, num_actions, num_filters, filter_size, hidden, lr)

    def _build_model(self, state_dim, num_actions, num_filters, filter_size, hidden, lr):
        """
        This method creates a neural network with two hidden fully connected layers and 20 neurons each. The output layer
        has #a neurons, where #a is the number of actions and has linear activation.
        Also creates its loss (mean squared loss) and its optimizer (e.g. Adam with a learning rate of 1e-4).
        """

        self.states_ = tf.placeholder(tf.float32, shape=[None, state_dim])
        self.actions_ = tf.placeholder(tf.int32, shape=[None])                  # Integer id of which action was selected
        self.targets_ = tf.placeholder(tf.float32,  shape=[None])               # The TD target value

        # network
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

        self.predictions = tf.layers.dense(output, num_actions, activation=tf.nn.relu)

        # Get the predictions for the chosen actions only
        batch_size = tf.shape(self.states_)[0]
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.targets_, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = self.optimizer.minimize(self.loss)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')

    def bias_variable(self, shape):
        initial = tf.zeros(shape=shape)
        return tf.Variable(initial)

    def predict(self, sess, states):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """
        prediction = sess.run(self.predictions, { self.states_: states })
        return prediction


    def update(self, sess, states, actions, targets):
        """
        Updates the weights of the neural network, based on its targets, its
        predictions, its loss and its optimizer.

        Args:
          sess: TensorFlow session.
          states: [current_state] or states of batch
          actions: [current_action] or actions of batch
          targets: [current_target] or targets of batch
        """
        feed_dict = { self.states_: states, self.targets_: targets, self.actions_: actions}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)

        return loss


class TargetNetwork(NeuralNetwork):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """
    def __init__(self, state_dim, num_actions, hidden=20, lr=1e-4, tau=0.01):
        super().__init__(state_dim, num_actions, hidden, lr)
        self.tau = tau
        self._associate = self._register_associate()

    def _register_associate(self):
        tf_vars = tf.trainable_variables()
        total_vars = len(tf_vars)
        op_holder = []
        for idx,var in enumerate(tf_vars[0:total_vars//2]):
            op_holder.append(tf_vars[idx+total_vars//2].assign(
              (var.value()*self.tau) + ((1-self.tau)*tf_vars[idx+total_vars//2].value())))
        return op_holder

    def update(self, sess):
        for op in self._associate:
          sess.run(op)
