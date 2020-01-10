#!/usr/bin/python

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
import math
import tensorflow as tf
import TensorFI as ti

FLAGS = None

def lambdaGen(res):
	"Return a lambda function to compute absolute difference with res if it's greater than threshold"
	threshold = 0.01
  	diff = lambda l: ( math.fabs(l - res) ) if (math.fabs(l - res) > threshold) else 0.0  
	return diff

def main(_):
  # Import data
  print("Importing data")
	
  mnist = input_data.read_data_sets(FLAGS.data_dir)

  print("Creating the model")
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.losses.sparse_softmax_cross_entropy on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  print("Training the model")
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  correct_prediction = tf.equal(tf.argmax(y, 1), y_) 
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # accuracy = tf.reduce_sum( tf.cast(correct_prediction, tf.int64) )	  
  
  # Test trained model
  print("Testing the model")
  correctOutput = sess.run(accuracy, feed_dict={x: mnist.test.images,
                        y_: mnist.test.labels})
  print("Correct output: ", correctOutput)
  	
  # instrument the session for fault injection (with fault injections disabled initially)
  fi = ti.TensorFI(sess, disableInjections = True, name = "MNIST-Injector", logLevel = 50)
  
  # Run the trained model (Must be run after the instrumentation)
  #  so that doInjections can invoke the fiLoop function to perform the injections
  correctOutput = sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels})

  # This is the function to generate a lambda function for comparing with the correct output
  comparisonFunction = lambdaGen(correctOutput)
  
  numRuns = 1100  

  numProcesses = 4

  # Initialize the statistics collection
  myStats = [ ]
  for i in range(numProcesses): 
	statName = "Stat" + str(i)
  	myStats.append( ti.FIStat(statName, "stats/" + statName ) )	

  # print("Correct output: ", correctOutput)
  # Launch the fault injection runs in parallel
  fi.pLaunch(numRuns, numProcesses,
		comparisonFunction,
		myStats,
		parallel = True,
		useProcesses = False )

  # fi.run( 10, lambdaGen, accuracy, 
  #		feed_dict={x: mnist.test.images, y_: mnist.test.labels},
  #		collectStats = myStats)

  # Print the fault injection statistics
  for i in range(numProcesses): 
  	print( "Thread " + str(i) + " statistics: " + myStats[i].getStats() )
	
  # Collate all the statistics across all the threads
  totalStats = ti.collateStats( myStats)
  print( "Overall Statistics: " + totalStats.getStats() )
	
  # Reset the injector - no more injections are possible	
  # fi.doneInjections()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
