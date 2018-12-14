import tensorflow as tf
import numpy as np

# TODO: add your Convolutional Neural Network for the CarRacing environment.

class NeuralNetwork():
    """
    Neural Network class based on TensorFlow.
    """
    def __init__(self, state_dim, num_actions, num_filters=32, filter_size=5, lr=1e-4, history_length=0):
        self._build_model(state_dim, num_actions, num_filters, filter_size, lr, history_length)

    def _build_model(self, state_dim, num_actions, num_filters, filter_size, lr, history_length):
        """
        This method creates a neural network with two hidden fully connected layers and 20 neurons each. The output layer
        has #a neurons, where #a is the number of actions and has linear activation.
        Also creates its loss (mean squared loss) and its optimizer (e.g. Adam with a learning rate of 1e-4).
        """
        self.state_length = history_length + 1

        self.states_ = tf.placeholder(tf.float32, shape=[None, 96, 96, self.state_length])
        self.actions_ = tf.placeholder(tf.int32, shape=[None])                  # Integer id of which action was selected
        self.targets_ = tf.placeholder(tf.float32,  shape=[None])               # The TD target value

        # network
        self.states_ = tf.convert_to_tensor(self.states_, dtype=tf.float32)
        #input_layer = tf.reshape(self.states_, tf.shape([None, 96, 96, self.state_length]))

        h_conv1 = tf.layers.conv2d(
            inputs=self.states_,
            filters=8,
            kernel_size=[7, 7],
            strides=[3, 3],
            padding="VALID",
            activation=tf.nn.relu)

        h_conv2 = tf.layers.conv2d(
            inputs=h_conv1,
            filters=16,
            kernel_size=[3, 3],
            strides=[2, 2],
            padding="VALID",
            activation=tf.nn.relu)

        h_conv2_flat = tf.contrib.layers.flatten(h_conv2)
        h_fc1 = tf.layers.dense(h_conv2_flat, 256, activation=tf.nn.relu)

        self.predictions = tf.layers.dense(h_fc1, num_actions, activation=None) # I THINK WE NEED TO ONE HOT

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
    def __init__(self, state_dim, num_actions, num_filters=32, filter_size=5, lr=1e-4, history_length=0, tau=0.01):
        super().__init__(state_dim, num_actions)
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
