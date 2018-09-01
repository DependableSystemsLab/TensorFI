#!/usr/bin/python

# Example2 from TensorFlow tutorial 

from __future__ import print_function
import tensorflow as tf
import TensorFI as ti

a = tf.placeholder(tf.float32, name="a")
b = tf.placeholder(tf.float32, name="b")
adder = tf.add(a, b, name="adder")  # Use this syntax for name
addTriple = 3 * adder

sess = tf.Session()

# Run the session with scalars and tensors
print( sess.run( addTriple, { a:3, b:4.5 } ) )
print( sess.run( addTriple, { a:[3,1], b:[4,5] } ) )

# Instrument the session
fi = ti.TensorFI(sess)

# Run the above session commands with fault injections
print( sess.run( addTriple, { a:3, b:4.5 } ) )
print( sess.run( addTriple, { a:[3,1], b:[4,5] } ) )

# Create a log for visualizing in TensorBoard
logs_path = "./logs"
logWriter = tf.summary.FileWriter( logs_path, sess.graph )

