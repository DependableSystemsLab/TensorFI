#!/usr/bin/python

""" Neural Network.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import TensorFI as ti
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import math
import sys

logPath = sys.argv[1]

# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_hidden_3 = 256 # 3rd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        
	# Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Training Finished!")

    # Add the fault injection code here to instrument the graph
    fi = ti.TensorFI(sess, name = "Perceptron", logLevel = 50, disableInjections = True)

    correctResult = sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels})

    print("Testing Accuracy:", correctResult)
    
    diffFunc = lambda x: math.fabs(x - correctResult)   	
  
    # Make the log files in TensorBoard	
    logs_path = "./logs"
    logWriter = tf.summary.FileWriter( logs_path, sess.graph )

    ###
    print("Accuracy (with no injections):", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
    orgAcy = accuracy.eval({X: mnist.test.images, Y: mnist.test.labels})
    fi.turnOnInjections()	
    print("Accuracy (with injections):", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
    fiAcy = accuracy.eval({X: mnist.test.images, Y: mnist.test.labels})
    ###

    with open(logPath, 'a') as of:
	of.write(`orgAcy` + " " + `fiAcy` + "\n")




    # Initialize the number of threads and injections
#    numThreads = 5 
#    numInjections = 100

    # Now start performing fault injections, and collect statistics
#    myStats = []
#    for i in range(numThreads):	
 #   	myStats.append( ti.FIStat("Perceptron") )	
   	
    # Launch the fault injections in parallel	
    #fi.pLaunch( numberOfInjections = numInjections, numberOfProcesses = numThreads, 
#		computeDiff = diffFunc, collectStatsList = myStats, timeout = 100)
 
    # Collate the statistics and print them	
 #   print( ti.collateStats(myStats).getStats() )
