#!/usr/bin/python

# set logging folder
from globals import TESTSUITE_DIR
logDir   = TESTSUITE_DIR + "/faultLogs/"
confFile = TESTSUITE_DIR + "/confFiles/operations_config.yaml"

# suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import yaml
yaml.warnings({'YAMLLoadWarning': False})

# for suppressing output
import sys

#imports
import tensorflow as tf
import TensorFI   as ti
import numpy      as np

# Get list of operations supported by TensorFI and inputgen functions for the inputs
from operations_inputgen import *

def run_test(suppress_out=False):

    if suppress_out:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    ops_passed = []
    ops_failed = []
    
    for op_type in inputgenMap:

        if op_type == "end_of_ops":
            continue # end of ops list, ignore

        sys.stdout.write("Testing op_type %s..." % op_type)
        sys.stdout.flush()

        # Create new graph context
        try:
            g = tf.Graph()
            graph_outputs = []
            with g.as_default():
                # generate inputs for op_type
                input_list = inputgenMap[op_type]()

                # loop through the generated inputs and create ops
                for input_set in input_list:
                    graph_outputs.extend(g.create_op(op_type, input_set).outputs)

            with tf.compat.v1.Session(graph=g) as sess:
                result_baseline = sess.run(graph_outputs)
                
                # instrument with TensorFI
                fi = ti.TensorFI(sess, disableInjections=True, name=op_type, configFileName= confFile, logDir=logDir)
                result_fi = sess.run(graph_outputs)
                
                # compare outputs
                passed = True
                for i,item in enumerate(result_fi):
                    if not np.array_equal(result_fi[i],result_baseline[i]):
                        temp_out = "\nFI element " + str(result_fi[i]) + " not equal to baseline " + str(result_baseline[i])
                        print temp_out
                        passed = False
                if passed:
                    sys.stdout.write("\rTesting op_type %s... Passed\n" % op_type)
                    sys.stdout.flush()
                    ops_passed.append(op_type)
                else:
                    print "\nTest FAILED for operation " + str(op_type)
                    print "Instrumented graph outputs not equal to original"
                    ops_failed.append(op_type)
        except Exception as e:
            print "\nTest FAILED for operation " + str(op_type)
            print "Exception thrown: " + str(e)
            ops_failed.append(op_type)
    
    if suppress_out:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    
    if len(ops_failed) > 0:
        return False, ops_passed, ops_failed    
    
    return True, ops_passed, ops_failed

if __name__ == "__main__":
    passed_bool, passed, failed = run_test()

    total_tests = len(passed) + len(failed)
    temp_out = "\n" + str(len(passed)) + "/" + str(total_tests) + " operations passed\n"
    print "\n\nOPERATION TEST RESULTS"
    print temp_out
    if len(failed) > 0:
        print "Failed operations: " + ', '.join(failed)