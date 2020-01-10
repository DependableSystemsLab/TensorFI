#!/usr/bin/python

# Example 4 from TensorFlow tutorial 

# set logging folder
from globals import TESTSUITE_DIR
logDir   = TESTSUITE_DIR + "/faultLogs/"
confFile = TESTSUITE_DIR + "/confFiles/injections_config.yaml" #specify the config file
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
	print "After initialization\tW: " + str(curr_W) + " b: " + str(curr_b) + " error: " + str(curr_error)

	# Iterate to train the model
	steps = 1000
	for i in range(steps):
		s.run( train, {x: x_train, y:y_train} )

	curr_W, curr_b, curr_error = s.run([W, b, error], {x: x_train, y: y_train})
	print "No injections\tW: " + str(curr_W) + " b: " + str(curr_b) + " error: " + str(curr_error)

	# Instrument the session
	fi = ti.TensorFI(s, logDir=logDir)

	# Create a log for visualizng in TensorBoard (during training)
	logs_path = "./logs"
	logWriter = tf.summary.FileWriter( logs_path, s.graph )

	# Turn off the injections during the first run
	fi.turnOffInjections()

	# Run the trained model without fault injections
	curr_W, curr_b, curr_error = s.run([W, b, error], {x: x_train, y: y_train})
	curr_W_A = curr_W
	curr_b_A = curr_b
	curr_error_A = curr_error
	print "Before injections\tW: " + str(curr_W) + " b: " + str(curr_b) + " error: " + str(curr_error)

	# Turn on the injections during running
	fi.turnOnInjections()

	# Run the trained model with the fault injected functions from the cached run
	curr_W, curr_b, curr_error = s.run(useCached = True)
	curr_W_B = curr_W
	curr_b_B = curr_b
	curr_error_B = curr_error
	print "After injections\tW: " + str(curr_W) + " b: " + str(curr_b) + " error: " + str(curr_error)
	
	if suppress_out:
		sys.stdout = sys.__stdout__
		sys.stderr = sys.__stderr__
	
	if curr_W_A == curr_W_B and curr_W_A == curr_W_B:
		passed_bool = True
	else:
		passed_bool = False

	return passed_bool


if __name__ == "__main__":
    passed_bool = run_test()

    print "\n\nTEST RESULTS"
    if passed_bool:
        print "Test passed"
    else:
        print "Test failed"
