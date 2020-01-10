#!/usr/bin/python

# Script to run the test suite
# Please include only stable and working tests in this file

# Check correct python version
import sys
if sys.version_info[0] == 2 and sys.version_info[1] >= 7:
    # version is 2.7+ but not 3
    pass
else:
    raise Exception("Please run with python 2.7+ (Python 3 is not supported)")

# Begin checking python package installations
sys.stdout.write("Checking for correct python packages...")
sys.stdout.flush()

# suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import yaml
yaml.warnings({'YAMLLoadWarning': False})

# Check that TensorFlow is installed
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except ImportError as e:
    print "\n\nTensorFlow is not installed properly, unable to import: " + str(e)
    exit()

# Check that PyYaml is installed
try:
    import yaml
except ImportError as e:
    print "\n\nPyYaml is not installed properly, unable to import: " + str(e)
    exit()

# Check that TensorFI is installed
try:
    import TensorFI
except ImportError as e:
    print "\n\nTensorFI is not installed properly, unable to import: " + str(e)
    exit()

# Check python package versions
if tuple(map(int, (tf.__version__.split("."))))[0] < 1:
    print "\n\nTensorFlow version must be 1.0 or newer"
    exit()
if tuple(map(int, (yaml.__version__.split("."))))[0] < 3:
    print "\n\nPyYaml version must be 3.0 or newer"
    exit()

sys.stdout.write("\rChecking for correct python packages... Passed\n")
sys.stdout.flush()

print "\nBeginning test cases...\n"
failed_tests = []

import operations_runTests
sys.stdout.write("Running operations_runTests test...")
sys.stdout.flush()
if operations_runTests.run_test(suppress_out=True)[0]:
    sys.stdout.write("\rRunning operations_runTests test... Passed\n")
    sys.stdout.flush()
else:
    failed_tests.append("operations_runTests.py")
    sys.stdout.write("\rRunning operations_runTests test... Failed\n")
    sys.stdout.flush()

import injections_alexnet_mnist
sys.stdout.write("Running injections_alexnet_mnist test...")
sys.stdout.flush()
if injections_alexnet_mnist.run_test(suppress_out=True):
    sys.stdout.write("\rRunning injections_alexnet_mnist test... Passed\n")
    sys.stdout.flush()
else:
    failed_tests.append("injections_alexnet_mnist.py")
    sys.stdout.write("\rRunning injections_alexnet_mnist test... Failed\n")
    sys.stdout.flush()

import injections_cnn_mnist
sys.stdout.write("Running injections_cnn_mnist test...")
sys.stdout.flush()
if injections_cnn_mnist.run_test(suppress_out=True):
    sys.stdout.write("\rRunning injections_cnn_mnist test... Passed\n")
    sys.stdout.flush()
else:
    failed_tests.append("injections_cnn_mnist.py")
    sys.stdout.write("\rRunning injections_cnn_mnist test... Failed\n")
    sys.stdout.flush()

import injections_linear_regression
sys.stdout.write("Running injections_linear_regression test...")
sys.stdout.flush()
if injections_linear_regression.run_test(suppress_out=True):
    sys.stdout.write("\rRunning injections_linear_regression test... Passed\n")
    sys.stdout.flush()
else:
    failed_tests.append("injections_linear_regression.py")
    sys.stdout.write("\rRunning injections_linear_regression test... Failed\n")
    sys.stdout.flush()

import injections_gradient
sys.stdout.write("Running injections_gradient test...")
sys.stdout.flush()
if injections_gradient.run_test(suppress_out=True):
    sys.stdout.write("\rRunning injections_gradient test... Passed\n")
    sys.stdout.flush()
else:
    failed_tests.append("injections_gradient.py")
    sys.stdout.write("\rRunning injections_gradient test... Failed\n")
    sys.stdout.flush()

import injections_logistic_regression
sys.stdout.write("Running injections_logistic_regression test...")
sys.stdout.flush()
if injections_logistic_regression.run_test(suppress_out=True):
    sys.stdout.write("\rRunning injections_logistic_regression test... Passed\n")
    sys.stdout.flush()
else:
    failed_tests.append("injections_logistic_regression.py")
    sys.stdout.write("\rRunning injections_logistic_regression test... Failed\n")
    sys.stdout.flush()

import injections_loss
sys.stdout.write("Running injections_loss test...")
sys.stdout.flush()
if injections_loss.run_test(suppress_out=True):
    sys.stdout.write("\rRunning injections_loss test... Passed\n")
    sys.stdout.flush()
else:
    failed_tests.append("injections_loss.py")
    sys.stdout.write("\rRunning injections_loss test... Failed\n")
    sys.stdout.flush()

# ------------------------------------
# add new test scripts above this line
# ------------------------------------ 

print "\n\nAll tests completed"
if len(failed_tests) == 0:
    print "All tests passed"
else:
    print "\nFailed tests: "
    for t in failed_tests:
        print t