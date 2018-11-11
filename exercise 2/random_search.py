import logging
logging.basicConfig(level=logging.WARNING)

import matplotlib
matplotlib.use('Agg')

import hpbandster.core.nameserver as hpns

from hpbandster.optimizers import RandomSearch

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import argparse

import tensorflow as tf
import numpy as np
from cnn_mnist import mnist


class MyWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test = mnist("./")

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    def compute(self, config, budget, **kwargs):
        """
        Evaluates the configuration on the defined budget and returns the validation performance.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train
        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        lr = config["learning_rate"]
        num_filters = config["num_filters"]
        batch_size = config["batch_size"]
        filter_size = config["filter_size"]
        epochs = budget

        #sess = tf.InteractiveSession()
        x_image = tf.placeholder(tf.float32, shape=[None,28,28,1], name='x')
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
        y_conv = tf.placeholder(tf.float32, shape=[None, 10])
        keep_prob = tf.placeholder(tf.float32, name='keep')


        W_conv1 = self.weight_variable([filter_size, filter_size, 1, num_filters])
        b_conv1 = self.bias_variable([num_filters])
        
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        W_conv2 = self.weight_variable([filter_size, filter_size, num_filters, num_filters])
        b_conv2 = self.bias_variable([num_filters])

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        W_fc1 = self.weight_variable([7 * 7 * num_filters, 128])
        b_fc1 = self.bias_variable([128])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*num_filters])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = self.weight_variable([128, 10])
        b_fc2 = self.bias_variable([10])

        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')  

        learning_curve = np.zeros(epochs)

        n_samples = self.x_train.shape[0]
        n_batches = n_samples // batch_size
        X_split = np.array_split(self.x_train, n_batches)
        Y_split = np.array_split(self.y_train, n_batches)    

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epochs):
                for b in range(int(n_batches)):
                    x_batch = X_split[b]
                    y_batch = Y_split[b]
                    
                    train_accuracy = accuracy.eval(feed_dict={x_image:x_batch, y_: y_batch, keep_prob: 1.0})
                    train_step.run(feed_dict={x_image: x_batch, y_: y_batch, keep_prob: 0.5})
                print("epoch %d, training accuracy %g"%(i, train_accuracy))
                learning_curve[i] = 1 - accuracy.eval(feed_dict={x_image: self.x_valid, y_: self.y_valid, keep_prob: 1.0})
        # TODO: train and validate your convolutional neural networks here
        validation_error = learning_curve[-1]
        # TODO: We minimize so make sure you return the validation error here
        return ({
            'loss': validation_error,  # this is the a mandatory field to run hyperband
            'info': {}  # can be used for any user-defined information - also mandatory
        })
        

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()

        learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=1e-4, upper=1e-1, default_value='1e-2', log=True)

        batch_size = CSH.UniformFloatHyperparameter('batch_size', lower=16, upper=128, default_value='64', log=True)

        num_filters = CSH.UniformIntegerHyperparameter('num_filters', lower=8, upper=64, default_value=16, log=True)

        filter_size = CSH.CategoricalHyperparameter('filter_size', [3, 5])

        cs.add_hyperparameters([learning_rate, batch_size, num_filters, filter_size])
        
        return cs


parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--budget', type=float,
                    help='Maximum budget used during the optimization, i.e the number of epochs.', default=6)
parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=50)
args = parser.parse_args()

# Step 1: Start a nameserver
# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine with the default port.
# The nameserver manages the concurrent running workers across all possible threads or clusternodes.
# Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()

# Step 2: Start a worker
# Now we can instantiate a worker, providing the mandatory information
# Besides the sleep_interval, we need to define the nameserver information and
# the same run_id as above. After that, we can start the worker in the background,
# where it will wait for incoming configurations to evaluate.
w = MyWorker(nameserver='127.0.0.1', run_id='example1')
w.run(background=True)

# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# Here, we run RandomSearch, but that is not essential.
# The run method will return the `Result` that contains all runs performed.

rs = RandomSearch(configspace=w.get_configspace(),
                  run_id='example1', nameserver='127.0.0.1',
                  min_budget=int(args.budget), max_budget=int(args.budget))
res = rs.run(n_iterations=args.n_iterations)

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
rs.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds information about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print('Best found configuration:', id2config[incumbent]['config'])


# Plots the performance of the best found validation error over time
all_runs = res.get_all_runs()
# Let's plot the observed losses grouped by budget,
import hpbandster.visualization as hpvis

hpvis.losses_over_time(all_runs)

import matplotlib.pyplot as plt
plt.savefig("random_search.png")

# TODO: retrain the best configuration (called incumbent) and compute the test error

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

x_train, y_train, x_valid, y_valid, x_test, y_test = mnist("./")

lr = id2config[incumbent]['config']["learning_rate"]
num_filters = int(id2config[incumbent]['config']["num_filters"])
batch_size = int(id2config[incumbent]['config']["batch_size"])
filter_size = int(id2config[incumbent]['config']["filter_size"])
epochs=6

print(lr)
print(num_filters)
print(batch_size)
print(filter_size)


x_image = tf.placeholder(tf.float32, shape=[None,28,28,1], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
y_conv = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32, name='keep')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


W_conv1 = weight_variable([filter_size, filter_size, 1, num_filters])
b_conv1 = bias_variable([num_filters])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([filter_size, filter_size, num_filters, num_filters])
b_conv2 = bias_variable([num_filters])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * num_filters, 128])
b_fc1 = bias_variable([128])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*num_filters])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([128, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')  

learning_curve = np.zeros(epochs)

n_samples = x_train.shape[0]
n_batches = n_samples // batch_size
X_split = np.array_split(x_train, n_batches)
Y_split = np.array_split(y_train, n_batches)    

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        for b in range(int(n_batches)):
            x_batch = X_split[b]
            y_batch = Y_split[b]
            
            train_step.run(feed_dict={x_image: x_batch, y_: y_batch, keep_prob: 0.5})

    print(accuracy.eval(feed_dict={x_image: x_test, y_: y_test, keep_prob: 1.0}))









