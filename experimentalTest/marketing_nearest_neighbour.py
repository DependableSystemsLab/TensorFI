#!/usr/bin/python
'''
A nearest neighbor learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf
import TensorFI as ti

import numpy, pandas
import preprocessing

import sys

logPath = sys.argv[1]
 

######
data = pandas.read_csv("./experimentalTest/marketing.csv")
data = preprocessing.cleanDataForClassification(data, "class")
 
labels = []
for d in data['class']:
    if int(d) == 0:
        labels.append([0,1])
    else:
        labels.append([1,0])
labels = pandas.DataFrame(labels).values
######
 
batch_xs = data.drop("class",axis=1).values
batch_ys = labels
Xtr = batch_xs[:5000]
Ytr = batch_ys[:5000]
Xte = batch_xs[20000:23000]
Yte = batch_ys[20000:23000]

# tf Graph Input
xtr = tf.placeholder("float", [None, 20])
xte = tf.placeholder("float", [20])

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
    fi = ti.TensorFI(sess, name = "NearestNeighbor", logLevel = 50, disableInjections = True)
    
    # loop over test data
    for i in range(len(Xte)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
            "True Class:", np.argmax(Yte[i]))
        # Calculate accuracy
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    orgAccuracy = accuracy
    print("Accuracy (Without FI):", accuracy)



    # Turn on TensorFI to inject faults in inference phase
    fi.turnOnInjections()
    accuracy = 0.
    for i in range(len(Xte)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
            "True Class:", np.argmax(Yte[i]))
        # Calculate accuracy
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print("Accuracy (Without FI):", orgAccuracy)
    print("Accuracy (With FI):", accuracy)


    # Make the log files in TensorBoard	
    logs_path = "./logs"
    logWriter = tf.summary.FileWriter( logs_path, sess.graph )

    with open(logPath, 'a') as of:
        of.write(`orgAccuracy` + "," + `accuracy` + "," + `(orgAccuracy - accuracy)` + '\n')

