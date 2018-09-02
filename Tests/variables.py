#!/usr/bin/python

# Example 3 from TensorFlow tutorial 

from __future__ import print_function
import tensorflow as tf
import TensorFI as ti

# Create 2 variables for keeping track of weights
W = tf.Variable([.3], dtype=tf.float32, name="W")
b = tf.Variable([-.3], dtype=tf.float32, name="b")

# Create a placeholder for inputs, and a linear model
x = tf.placeholder(tf.float32)
linear_model = W*x + b

# Create a session, initialize variables and run the linear model
s = tf.Session()
init = tf.global_variables_initializer()
print("Initial : ", s.run(init))
print("Linear Model : ", s.run(linear_model, { x: [1, 2, 3, 4] }))

# Instrument the FI session 
fi = ti.TensorFI(s)

# Create a log for visualizing in TensorBoard
logs_path = "./logs"
logWriter = tf.summary.FileWriter( logs_path, s.graph )

# Create a session, initialize variables and run the linear model, with faults this time
init = tf.global_variables_initializer()
print("Initial : ", s.run(init))
print("Linear Model : ", s.run(linear_model, { x: [1, 2, 3, 4] }))

