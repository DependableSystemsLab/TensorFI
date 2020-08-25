#/usr/bin/python

# Example1 from TensorFlow tutorial 

from __future__ import print_function
import sys

import tensorflow as tf
import TensorFI as ti

node1 = tf.constant(3, dtype=tf.float64)
node2 = tf.constant(4, dtype=tf.float64)

print("Node1 = ", node1)
print("Node 2 = ", node2)
node3 = tf.add(node1, node2, name = "add1")
print("Node3 = ", node3)

s = tf.Session()

# Run it first
res1 = s.run([ node3 ])
print("res1 = ", res1)

