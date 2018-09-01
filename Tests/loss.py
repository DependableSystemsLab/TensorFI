#!/usr/bin/python

# Example 4 from TensorFlow tutorial 

from __future__ import print_function
import sys
import tensorflow as tf
sys.path.append("/data/gpli/tensorfi/TensorFI/TensorFI/")
import TensorFI as ti

# Create 2 variables for keeping track of weights
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Create a placeholder for inputs, and a linear model
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W*x + b
squared_deltas = tf.square( linear_model - y )
loss = tf.reduce_sum(squared_deltas)

init = tf.global_variables_initializer()

# Create a session, initialize variables and run the linear model
s = tf.Session()
print("Initial : ", s.run(init))
print("Linear Model : ", s.run(linear_model, { x: [1, 2, 3, 4] }))
print("Loss Function : ", s.run(loss, { x: [1, 2, 3, 4], y: [0, -1, -2, -3] }))

# Instrument the session
fi = ti.TensorFI(s)

# Create a log for visualizng in TensorBoard
logs_path = "./logs"
logWriter = tf.summary.FileWriter( logs_path, s.graph )

# Create a session, initialize variables and run the linear model
print("Initial : ", s.run(init))
print("Linear Model : ", s.run(linear_model, { x: [1, 2, 3, 4] }))
print("Loss Function : ", s.run(loss, { x: [1, 2, 3, 4], y: [0, -1, -2, -3] }))
