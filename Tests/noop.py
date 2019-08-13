#/usr/bin/python

# Example 0 - Dummy NoOp operation

from __future__ import print_function
import sys

import tensorflow as tf
import TensorFI as ti

node = tf.no_op()

print("Node = ", node)

s = tf.Session()

# Run it first
res1 = s.run([ node ])
print("res1 = ", res1)

# Instrument the FI session 
fi = ti.TensorFI(s, logLevel = 0)

# Create a log for visualizng in TensorBoard
logs_path = "./logs"
logWriter = tf.summary.FileWriter( logs_path, s.graph )

# Run it again with fault injection enabled
res2 = s.run([ node ])
print("res2 = ", res2)

