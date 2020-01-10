#!/usr/bin/python
'''
A nearest neighbor learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

# NOTE: please define how this test "passes" (at the moment, passed_bool is always True)
# once this is done, include in runAll.py

import numpy as np

# set logging folder
from globals import TESTSUITE_DIR
logDir   = TESTSUITE_DIR + "/faultLogs/"
confFile = TESTSUITE_DIR + "/confFiles/***_config.yaml" #specify the config file
# ***the call to TensorFI to instrument the model must include these directories, e.g.,  fi = ti.TensorFI(..., configFileName= confFile, logDir=logDir)

# suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import yaml
yaml.warnings({'YAMLLoadWarning': False})

import tensorflow as tf
import TensorFI as ti
import sys

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def run_test(suppress_out=False):

    if suppress_out:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    
    # Import MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # In this example, we limit mnist data
    Xtr, Ytr = mnist.train.next_batch(5000) #5000 for training (nn candidates)
    Xte, Yte = mnist.test.next_batch(200) #200 for testing

    # tf Graph Input
    xtr = tf.placeholder("float", [None, 784])
    xte = tf.placeholder("float", [784])

    # Nearest Neighbor calculation using L1 Distance
    # Calculate L1 Distance
    distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
    # Prediction: Get min distance index (Nearest neighbor)
    pred = tf.arg_min(distance, 0)

    accuracy = 0.

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Add the fault injection code here to instrument the graph
        # We start injecting the fault right away here unlike earlier
        fi = ti.TensorFI(sess, name = "NearestNeighbor", logLevel = 50, logDir=logDir)
        
        # loop over test data
        for i in range(len(Xte)):
            # Get nearest neighbor
            nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
            # Get nearest neighbor class label and compare it to its true label
            print "Test " + str(i) + ", Prediction: " + str(np.argmax(Ytr[nn_index])) + ", True Class: " + str(np.argmax(Yte[i]))
            # Calculate accuracy
            if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
                accuracy += 1./len(Xte)
        print "Accuracy:" + str(accuracy)

        # Make the log files in TensorBoard	
        logs_path = "./logs"
        logWriter = tf.summary.FileWriter( logs_path, sess.graph )

    passed_bool = True

    if suppress_out:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    
    return passed_bool # set this based on the test requirements (True if the test passed, False otherwise)

if __name__ == "__main__":
    passed_bool = run_test()

    print "\n\nTEST RESULTS"
    if passed_bool:
        print "Test passed"
    else:
        print "Test failed"
