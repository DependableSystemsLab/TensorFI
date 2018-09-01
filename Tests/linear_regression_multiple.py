#!/usr/bin/python
'''
A linear regression learning algorithm example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
import TensorFI as ti
rng = np.random

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
accuracies_ok= []
accuracies_faulty = []
num_trials = 2000

with tf.Session() as sess:

	# Run the initializer
	sess.run(init)

     	# Fit all training data
     	for epoch in range(training_epochs):
         	for (x, y) in zip(train_X, train_Y):
     			sess.run(optimizer, feed_dict={X: x, Y: y})

            	# Display logs per epoch step
            	if (epoch+1) % display_step == 0:
                	c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
                	print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                      		"W=", sess.run(W), "b=", sess.run(b))

        print("Optimization Finished!")

        training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

        # Testing example, as requested (Issue #2)
        test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
        test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

        print("Testing... (Mean square loss Comparison)")
        accuracy = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        testing_cost = sess.run(
            		accuracy,
	    		feed_dict={X: test_X, Y: test_Y})  # same function as cost above
        print("Testing cost=", testing_cost)
        print("Absolute mean square loss difference:", abs(
            training_cost - testing_cost))

        # Calculate accuracy (before fault injections)
        print("Accuracy:", sess.run(accuracy, feed_dict={X: test_X, Y: test_Y}))
    
        # Instrument the graph for fault injection 
        fi = ti.TensorFI(sess, name = "linearReg", logLevel = 50, disableInjections = True)
    
        # Calculate accuracy (with no fault injections)
        accuracy_ok = sess.run(accuracy, feed_dict={X: test_X, Y: test_Y})
        print("Accuracy (no injections):", accuracy_ok)
        accuracies_ok.append(accuracy_ok)

	# Now do the fault injections
        fi.turnOnInjections()
	for trial in range(num_trials):
        	
		# Calculate accuracy (with fault injections)
        	accuracy_faulty = sess.run(accuracy, feed_dict={X: test_X, Y: test_Y})
        	
		print("Accuracy (with injections):", accuracy_faulty)
        	accuracies_faulty.append(accuracy_faulty)

# Print the statistics
print("No faults: ",np.average(accuracies_ok), "+/-", np.std(accuracies_ok)/np.sqrt(num_trials))
print("Faults: ",np.average(accuracies_faulty), "+/-", np.std(accuracies_faulty)/np.sqrt(num_trials))              


