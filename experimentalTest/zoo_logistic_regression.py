'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy, pandas
import preprocessing

import TensorFI as ti

import sys

logPath = sys.argv[1]


# Parameters
learning_rate = 0.05
training_epochs = 1000
batch_size = 100
display_step = 1

######
data = pandas.read_csv("./experimentalTest/zoo.csv")
data = preprocessing.cleanDataForClassification(data, "class")
data = data.drop("Name",axis=1)

labels = []
for d in data['class']:
    if int(d) == 1:
	labels.append([0,0,0,0,0,0,1])
    if int(d) == 2:
	labels.append([0,0,0,0,0,1,0])
    if int(d) == 3:
	labels.append([0,0,0,0,1,0,0])
    if int(d) == 4:
	labels.append([0,0,0,1,0,0,0])
    if int(d) == 5:
	labels.append([0,0,1,0,0,0,0])
    if int(d) == 6:
	labels.append([0,1,0,0,0,0,0])
    if int(d) == 7:
	labels.append([1,0,0,0,0,0,0])
labels = pandas.DataFrame(labels).values
######

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 16]) # 16 features in zoo dataset
y = tf.placeholder(tf.float32, [None, 7]) # 7 classes of animals

# Set model weights
W = tf.Variable(tf.zeros([16, 7]))
b = tf.Variable(tf.zeros([7]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.

        batch_xs = data.drop("class",axis=1).values
	batch_ys = labels
	    
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs[:50],
                                                          y: labels[:50]})
        # Compute average loss
        avg_cost = c

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy (Without FI):", accuracy.eval({x: batch_xs[50:], y: labels[50:]}))
    orgAcy = accuracy.eval({x: batch_xs[50:], y: labels[50:]})



    # Turn on TensorFI to inject faults in inference phase
    fi = ti.TensorFI(sess, name = "logistReg", logLevel = 30, disableInjections = True)
    fi.turnOnInjections()	
    print("Accuracy (With FI):", accuracy.eval({x: batch_xs[50:], y: labels[50:]}))
    fiAcy = accuracy.eval({x: batch_xs[50:], y: labels[50:]})


    with open(logPath, 'a') as of:
        of.write(`orgAcy` + "," + `fiAcy` + "," + `(orgAcy - fiAcy)` + '\n')

