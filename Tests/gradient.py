#!/usr/bin/python

# Example 4 from TensorFlow tutorial 

from __future__ import print_function
import tensorflow as tf
import TensorFI as ti

# Create 2 variables for keeping track of weights
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Create a placeholder for inputs, and a linear model
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W*x + b

# Calculate the error as the sum of square of the dviations from the linear model 
squared_deltas = tf.square( linear_model - y )
error = tf.reduce_sum(squared_deltas)

# Initialize a gradient descent optimizer to minimize errors
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(error)

# Training data for x and y
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# Create a session, initialize variables
s = tf.Session()
init = tf.global_variables_initializer()
s.run(init)

# Run the initial model
curr_W, curr_b, curr_error = s.run([W, b, error], {x: x_train, y: y_train})
print("After initialization\tW: %s b: %s error: %s"%(curr_W, curr_b, curr_error))

# Iterate to train the model
steps = 1000
for i in range(steps):
	s.run( train, {x: x_train, y:y_train} )

curr_W, curr_b, curr_error = s.run([W, b, error], {x: x_train, y: y_train})
print("No injections\tW: %s b: %s error: %s"%(curr_W, curr_b, curr_error))

# Instrument the session
fi = ti.TensorFI(s)

# Create a log for visualizng in TensorBoard (during training)
logs_path = "./logs"
logWriter = tf.summary.FileWriter( logs_path, s.graph )

# Turn off the injections during the first run
fi.turnOffInjections()

# Run the trained model without fault injections
curr_W, curr_b, curr_error = s.run([W, b, error], {x: x_train, y: y_train})
print("Before injections\tW: %s b: %s error: %s"%(curr_W, curr_b, curr_error))

# Turn on the injections during running
fi.turnOnInjections()

# Run the trained model with the fault injected functions from the cached run
curr_W, curr_b, curr_error = s.run(useCached = True)
print("After injections\tW: %s b: %s error: %s"%(curr_W, curr_b, curr_error))
