#!/usr/bin/python

# This is an empty test script that follows the format for proper inclusion in the runAll.py main script
# To include a new test in the suite, add the following lines of code to runAll.py and
# change all instances of REFERENCE_empty_test_script to the file name
#      
#  import REFERENCE_empty_test_script
#  sys.stdout.write("Running REFERENCE_empty_test_script test...")
#  sys.stdout.flush()
#  if REFERENCE_empty_test_script.run_test(suppress_out=True):
#      sys.stdout.write("\rRunning REFERENCE_empty_test_script test... Passed\n")
#      sys.stdout.flush()
#  else:
#      failed_tests.append("REFERENCE_empty_test_script.py")
#      sys.stdout.write("\rRunning REFERENCE_empty_test_script test... Failed\n")
#      sys.stdout.flush()
#
# The main code for the test should go inside the run_test() function, as indicated

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

    # ************************
    # ************************
    # MAIN TEST CODE GOES HERE
    # ************************
    # ************************

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
