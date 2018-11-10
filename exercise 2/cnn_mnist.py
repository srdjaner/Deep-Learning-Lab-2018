from __future__ import print_function

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')

import tempfile
import argparse
import gzip
import json
import os
import pickle
import matplotlib.pyplot as plt

import numpy as np

SAVE_PATH = 'C:/Users/Srdjan/Desktop/Laboratory AIS/!dl-lab-2018-master/exercise2'

sess = tf.InteractiveSession()

def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 28, 28, 1)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 28, 28, 1)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 28, 28, 1)
    train_y = train_y.astype('int32')
    print('... done loading data')
    return train_x, one_hot(train_y), valid_x, one_hot(valid_y), test_x, one_hot(test_y)

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

def train_and_validate(x_train, y_train, x_valid, y_valid, num_epochs, lr, num_filters, batch_size, filter_size=3, dropout=0.6):
    # TODO: train and validate your convolutional neural networks with the provided data and hyperparameters
    x_image = tf.placeholder(tf.float32, shape=[None,28,28,1], name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
    keep_prob = tf.placeholder(tf.float32, name='keep')    #for dropout
    y_conv = tf.placeholder(tf.float32, shape=[None, 10], name='y_conv')
    
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

    saver = tf.train.Saver()

    learning_curve = np.zeros(num_epochs)

    n_samples = x_train.shape[0]
    n_batches = n_samples // batch_size
    X_split = np.array_split(x_train, n_batches)
    Y_split = np.array_split(y_train, n_batches)    
    
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        for i in range(num_epochs):
            for b in range(n_batches):
                x_batch = X_split[b]
                y_batch = Y_split[b]
                
                train_accuracy = accuracy.eval(feed_dict={x_image:x_batch, y_: y_batch, keep_prob: 1.0})
                train_step.run(feed_dict={x_image: x_batch, y_: y_batch, keep_prob: dropout})
            print("epoch %d, training accuracy %g"%(i, train_accuracy))
            learning_curve[i] = 1 - accuracy.eval(feed_dict={x_image: x_valid, y_: y_valid, keep_prob: 1.0})
        temp = tempfile.NamedTemporaryFile()
        save_path = saver.save(sess, SAVE_PATH + "/models/" + temp.name + ".ckpt")

    return learning_curve, save_path


def test(x_test, y_test, model):
    # TODO: test your network here by evaluating it on the test data
    
    graph = tf.get_default_graph()
    
    saver = tf.train.Saver()
    saver.restore(sess, model)
    
    accuracy = graph.get_tensor_by_name('accuracy:0')
    x_image = graph.get_tensor_by_name('x:0')
    y_ = graph.get_tensor_by_name('y_:0')
    y_conv = graph.get_tensor_by_name('y_conv:0')
    keep_prob = graph.get_tensor_by_name('keep:0')
    print("Test accuracy is: %g" %accuracy.eval(feed_dict={x_image: x_test, y_: y_test, keep_prob: 1.0}))
    return 1-accuracy.eval(feed_dict={x_image: x_test, y_: y_test, keep_prob: 1.0})


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="./", type=str, nargs="?",
                        help="Path where the results will be stored")
    parser.add_argument("--input_path", default="./", type=str, nargs="?",
                        help="Path where the data is located. If the data is not available it will be downloaded first")
    parser.add_argument("--learning_rate", default=0.001, type=float, nargs="?", help="Learning rate for SGD")
    parser.add_argument("--num_filters", default=16, type=int, nargs="?",
                        help="The number of filters for each convolution layer")
    parser.add_argument("--batch_size", default=128, type=int, nargs="?", help="Batch size for SGD")
    parser.add_argument("--epochs", default=12, type=int, nargs="?",
                        help="Determines how many epochs the network will be trained")
    parser.add_argument("--run_id", default=0, type=int, nargs="?",
                        help="Helps to identify different runs of an experiments")

    args = parser.parse_args()

    # hyperparameters
    lr = args.learning_rate
    num_filters = args.num_filters
    batch_size = args.batch_size
    epochs = args.epochs

    # train and test convolutional neural network
    x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)
    
    learning_curve, model = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters, batch_size, 3)
    #t = range(0, epochs)
    plt.plot(learning_curve, label="test1")
    plt.ylabel('Validation error')
    plt.xlabel('Epochs')
    plt.title('Initial parameters')

    plt.savefig(SAVE_PATH + '/testing.png')
    plt.close()

    test_error = test(x_test, y_test, model)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["lr"] = lr
    results["num_filters"] = num_filters
    results["batch_size"] = batch_size
    results["learning_curve"] = learning_curve.tolist()
    results["test_error"] = test_error.tolist()

    path = os.path.join(args.output_path, "results")
    os.makedirs(path, exist_ok=True)

    fname = os.path.join(path, "results_run_%d.json" % args.run_id)

    fh = open(fname, "w")
    json.dump(results, fh)
    fh.close()


    # learning rates
    rates = [0.1, 0.01, 0.001, 0.0001]
    for i in range(0, len(rates)):
        learning_curve, model = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, rates[i], num_filters, batch_size)
        test_error = test(x_test, y_test, model)

        results = dict()
        results["lr"] = rates[i]
        results["filter_size"] = 3
        results["num_filters"] = num_filters
        results["batch_size"] = batch_size
        results["learning_curve"] = learning_curve.tolist()
        results["test_error"] = test_error.tolist()

        path_lr = os.path.join(args.output_path, "results_learning_rates")
        os.makedirs(path_lr, exist_ok=True)

        fname = os.path.join(path_lr, "results_lr_%d.json" % i)

        fh = open(fname, "w")
        json.dump(results, fh)
        fh.close()
        
        plt.plot(learning_curve, label=str(rates[i]))

    plt.ylabel('Validation error')
    plt.xlabel('Epochs')
    plt.title('Learning rates testing')
    plt.legend()
    plt.savefig(SAVE_PATH + '/learning_rates.png')
    plt.close()


    # filter sizes
    sizes = [1, 3, 5, 7]
    for i in range(0, len(sizes)):
        learning_curve, model = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters, batch_size, sizes[i])
        test_error = test(x_test, y_test, model)

        results = dict()
        results["lr"] = lr
        results["filter_size"] = sizes[i]
        results["num_filters"] = num_filters
        results["batch_size"] = batch_size
        results["learning_curve"] = learning_curve.tolist()
        results["test_error"] = test_error.tolist()
        
        
        path_fs = os.path.join(args.output_path, "results_filter_sizes")
        os.makedirs(path_fs, exist_ok=True)

        fname = os.path.join(path_fs, "results_fs_%d.json" % i)

        fh = open(fname, "w")
        json.dump(results, fh)
        fh.close()

        plt.plot(learning_curve, label=str(sizes[i]))

    plt.ylabel('Validation error')
    plt.xlabel('Epochs')
    plt.title('Filter sizes testing')
    plt.legend()
    plt.savefig(SAVE_PATH + '/filter_sizes.png')
    plt.close()

        
