#!/usr/bin/python

# set logging folder
from globals import TESTSUITE_DIR
logDir = TESTSUITE_DIR + "/faultLogs/"

# suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# for suppressing output
import sys

import tensorflow as tf
import TensorFI   as ti
import numpy      as np

# Get list of operations supported by TensorFI by importing the operation_list.py
from operation_list import op_list

def run_test(suppress_out=False):

    if suppress_out:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    ops_passed = []
    ops_failed = []
    
    for op in op_list:
        op_type     = op[1]
        input_types = op[2]

        # Create new graph context
        g = tf.Graph() 
        graph_outputs = []
        with g.as_default():
            # Create graph tensors and ops, per input type
            for input_type in input_types:
                if input_type == "float":
                    float_a   = tf.constant([[1.22,3.14],[0.9,8.37]],dtype=tf.float32)
                    float_b   = tf.constant([[0.90,8.27],[1.32,3.54]],dtype=tf.float32)
                    float_c   = tf.constant([[-1.2,-3.4],[3.4,-5.2]],dtype=tf.float32)
                    float_d   = tf.constant([[7.3,-5.34],[-8.7,-0.1]],dtype=tf.float32)
                    float_op1 = g.create_op(op_type, [float_a, float_b])
                    float_op2 = g.create_op(op_type, [float_c, float_d])
                    graph_outputs.extend(float_op1.outputs)
                    graph_outputs.extend(float_op2.outputs)
                elif input_type == "int":
                    int_a     = tf.constant([[2,4],[37,51]],dtype=tf.int32)
                    int_b     = tf.constant([[9,7],[54,6]],dtype=tf.int32)
                    int_c     = tf.constant([[-2,-3],[4,9]],dtype=tf.int32)
                    int_d     = tf.constant([[3,-4],[-7,-1]],dtype=tf.int32)
                    int_op1   = g.create_op(op_type, [int_a, int_b])
                    int_op2   = g.create_op(op_type, [int_c, int_d])
                    graph_outputs.extend(int_op1.outputs)
                    graph_outputs.extend(int_op2.outputs)  
                elif input_type == "bool":
                    bool_a    = tf.constant([[True, True],[False, False]],dtype=tf.bool)
                    bool_b    = tf.constant([[True, False],[True, False]],dtype=tf.bool)
                    bool_c    = tf.constant([[True, False],[False, True]],dtype=tf.bool)
                    bool_d    = tf.constant([[True, True],[True, False]],dtype=tf.bool)
                    bool_op1  = g.create_op(op_type, [bool_a, bool_b])
                    bool_op2  = g.create_op(op_type, [bool_c, bool_d])
                    graph_outputs.extend(bool_op1.outputs)
                    graph_outputs.extend(bool_op2.outputs)
                elif input_type == "string":
                    str_a     = tf.constant([["DkwGThU4NX", "DSsbQWMOBa"],["3ShrtGpqpA", "5iGaYXtI3H"]])
                    str_b     = tf.constant([["lWEopVE8WY", "tFDZ6HPaXt"],["YCDxgt3kga", "08ny3rpzMN"]])
                    str_c     = tf.constant([["syVsesW18k", "THBbaZwNMs"],["RJ2WHetuQY", "BCNgcvdsxv"]])
                    str_d     = tf.constant([["1mzrE4aSbv", "7iBIxjlBcD"],["oyKKOwSkSJ", "CBz91f8FWp"]])
                    str_op1   = g.create_op(op_type, [str_a, str_b])
                    str_op2   = g.create_op(op_type, [str_c, str_d])
                    graph_outputs.extend(str_op1.outputs)
                    graph_outputs.extend(str_op2.outputs)
                elif input_type == "complex":
                    compl_a   = tf.complex(tf.constant([[3.4,5.76],[0.9,8.37],[4.23,1.43]],dtype=tf.float32),tf.constant([[-5.4,-2.2],[-4.2,-0.1],[-4.4,2.5]],dtype=tf.float32))
                    compl_b   = tf.complex(tf.constant([[8.7,6.25],[1.32,3.54],[17.8,9.21]],dtype=tf.float32),tf.constant([[0.9,6.25],[1.32,3.54],[15.8,9.21]],dtype=tf.float32))
                    compl_c   = tf.complex(tf.constant([[-3.4,-5.6],[3.4,-5.2],[9.7,-4.2]],dtype=tf.float32),tf.constant([[-1.2,-3.1],[3.4,-5.2],[9.5,-2.5]],dtype=tf.float32))
                    compl_d   = tf.complex(tf.constant([[-5.4,-5.5],[-8.7,-0.1],[-4.4,15.0]],dtype=tf.float32),tf.constant([[1.2,3.4],[0.9,8.31],[4.23,2.71]],dtype=tf.float32))
                    compl_op1 = g.create_op(op_type, [compl_a, compl_b])
                    compl_op2 = g.create_op(op_type, [compl_c, compl_d])
                    graph_outputs.extend(compl_op1.outputs)
                    graph_outputs.extend(compl_op2.outputs)

        with tf.compat.v1.Session(graph=g) as sess:
            print "OP-TYPE: " + op_type
            result_baseline = sess.run(graph_outputs)
            
            # instrument with TensorFI
            fi = ti.TensorFI(sess, disableInjections=True, name=op_type, configFileName="confFiles/regression_operators.yaml", logDir=logDir)
            result_fi = sess.run(graph_outputs)
            
            # compare outputs
            passed = True
            for i,item in enumerate(result_fi):
                if not np.array_equal(result_fi[i],result_baseline[i]):
                    temp_out = "FI element " + str(result_fi[i]) + " not equal to baseline " + str(result_baseline[i])
                    print temp_out
                    passed = False
            if passed:
                print "Test passed for operation " + str(op[0])
                ops_passed.append(op[0])
            else:
                print "Test FAILED for operation " + str(op[0])
                ops_failed.append(op[0])
    
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
    print "\n\nOPERATOR TEST RESULTS"
    print temp_out
    if len(failed) > 0:
        print "Failed operations: " + ', '.join(failed)