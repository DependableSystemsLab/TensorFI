#!/usr/bin/python

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
    init_nofi = s.run(init)
    print "Initial : " + str(init_nofi)
    model_nofi = s.run(linear_model, { x: [1, 2, 3, 4] })
    print "Linear Model : " + str(model_nofi)
    loss_nofi = s.run(loss, { x: [1, 2, 3, 4], y: [0, -1, -2, -3] })
    print "Loss Function : " + str(loss_nofi)

    # Instrument the session
    fi = ti.TensorFI(s, logDir=logDir)

    # Create a log for visualizng in TensorBoard
    logs_path = "./logs"
    logWriter = tf.summary.FileWriter( logs_path, s.graph )

    # initialize variables and run the linear model
    init_fi = s.run(init)
    print "Initial : " + str(init_fi)
    model_fi = s.run(linear_model, { x: [1, 2, 3, 4] })
    print "Linear Model : " + str(model_fi)
    loss_fi = s.run(loss, { x: [1, 2, 3, 4], y: [0, -1, -2, -3] })
    print "Loss Function : " + str(loss_fi)

    if loss_nofi == loss_fi:
        passed_bool = True
    else:
        passed_bool = False

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
