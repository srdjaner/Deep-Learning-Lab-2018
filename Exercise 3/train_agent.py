from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

from random import randint
from sklearn.utils import shuffle

from model import Model
from utils import *
from tensorboard_evaluation import Evaluation
import tensorflow as tf
from datetime import datetime

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file, 'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid

def preprocessing(X_train1, X_train2, y_train, X_valid, y_valid, history_length=1):
    print("... preprocessing")
    y_temp1 = np.zeros(y_train.shape[0], dtype=np.int8)
    y_temp2 = np.zeros(y_valid.shape[0], dtype=np.int8)


    X_train1 = rgb2gray(X_train1)
    X_train2 = rgb2gray(X_train2)
    X_train = np.concatenate((X_train1, X_train2),axis=0)
    X_valid = rgb2gray(X_valid)
    
    for i in range(0, X_train.shape[0]):
        y_temp1[i] = action_to_id(y_train[i])

    for i in range(0, X_valid.shape[0]):
        y_temp2[i] = action_to_id(y_valid[i])

    y_valid = one_hot(y_temp2)
    
    return X_train, y_temp1, X_valid, y_valid


def sample_minibatch(X, y, index, batch_size, history_length=1):
    b = randint(0, index.shape[0]-batch_size-1)
    batch_index = index[b :b + batch_size]
    X_batch = np.zeros((batch_size, X.shape[1], X.shape[2], history_length))
    y_batch = np.zeros(batch_size)
    for i in range(history_length):
        X_batch[:, :, :, i] = X[batch_index+i]
    y_batch[:] = y[batch_index+history_length-1]

    return X_batch, y_batch

def make_history(X, y, history):
    X_batch = np.zeros((X.shape[0]-history, X.shape[1], X.shape[2], history))
    Y_batch = y[history:]
    for i in range(history):
        X_batch[:, :, :, i] = np.reshape(X[i:X.shape[0]-history+i], (X.shape[0]-history, 96, 96))

    return X_batch, Y_batch

def uniform_sampling(X_train, y_train_id_n, number):

    n = X_train.shape[0]
    weights = np.zeros(n)
    left_indices = y_train_id_n == 1
    weights[left_indices] = n / np.sum(left_indices)
    right_indices = y_train_id_n == 2
    weights[right_indices] = n / np.sum(right_indices)
    straight_indices = y_train_id_n == 0
    weights[straight_indices] = n / np.sum(straight_indices)
    acce_indices = y_train_id_n == 3
    weights[acce_indices] = n / np.sum(acce_indices)
    brake_indices = y_train_id_n == 4
    weights[brake_indices] = n / np.sum(brake_indices)

    weights = weights / np.sum(weights)
    samples_indices = np.random.choice(np.arange(n), number, replace=False, p=weights)

    return samples_indices


def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr ,num_filters, filter_size, history_length, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")

    agent = Model(filter_size=filter_size, num_filters=num_filters, learning_rate=lr, history_length=history_length)
    agent.sess.run(tf.global_variables_initializer())
    tensorboard_eval = Evaluation(tensorboard_dir)
    tf.reset_default_graph()

    #X_train, y_train = shuffle(X_train, y_train, random_state = 0) # but if i shuffle I loose history!
    X_valid = np.reshape(X_valid, (X_valid.shape[0], 96, 96, 1))

    n_samples = X_valid.shape[0]
    n_batches = n_samples // batch_size
    X_split = np.array_split(X_valid, n_batches)
    Y_split = np.array_split(y_valid, n_batches)
    for i in range(n_batches):
        X_split[i], Y_split[i] = make_history(X_split[i], Y_split[i], history_length)

    total = n_minibatches//1000

    train = np.zeros(total)
    valid = np.zeros(total)

    index = uniform_sampling(X_train[history_length-1:], y_train[history_length-1:], 25000)
    
    for i in range(n_minibatches):
        x, y = sample_minibatch(X_train, y_train, index, batch_size, history_length)
        y_hot = one_hot(y)
        agent.sess.run([agent.train_step, agent.cross_entropy], feed_dict={agent.x_image: x, agent.y_: y_hot})

        if ((i+1)%1000==0):
            step=(i+1)//1000
            index = uniform_sampling(X_train[history_length-1:], y_train[history_length-1:], 25000)
            print("Training step: " + str(step) + " of " + str(total))
            train[step-1] = agent.accuracy.eval(session=agent.sess, feed_dict={agent.x_image:x, agent.y_: y_hot})
            for j in range(n_batches):
                valid[step-1] += agent.accuracy.eval(session=agent.sess, feed_dict={agent.x_image:X_split[j], agent.y_: Y_split[j]})
            valid[step-1] = valid[step-1]/n_batches
            print("Train accuracy: "+ str(train[step-1]))
            print("Test accuracy: " + str(valid[step-1]))
            eval_dict = {"train":train[step-1], "valid":valid[step-1]}
            tensorboard_eval.write_episode_data(step, eval_dict)


   
    # TODO: save your agent
    model_dir = agent.save(os.path.join(model_dir, "agent.ckpt"))
    print("Model saved in file: %s" % model_dir)
    agent.sess.close()

if __name__ == "__main__":

    history_length = 5

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    X_train1=X_train[:20000]
    X_train2=X_train[20000:]
    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train1, X_train2, y_train, X_valid, y_valid, history_length)
    
    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, n_minibatches=10000, batch_size=64, lr=0.0001, num_filters=32, filter_size=5, history_length=history_length)
    datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
