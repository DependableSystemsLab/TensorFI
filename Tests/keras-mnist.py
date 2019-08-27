#!/usr/bin/python
# MNIST dataset recognition using Keras - example taken from the Keras tutorial
#  https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html#calling-keras-layers-on-tensorflow-tensors

from __future__ import print_function
import sys
import imp

import tensorflow as tf
import TensorFI as ti

# Remove deprecated warnings from TensorFlow
tf.logging.set_verbosity(tf.logging.FATAL)
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

# this placeholder will contain our input digits, as flat vectors
img = tf.placeholder(tf.float32, shape=(None, 784))

from keras.layers import Dense

# Keras layers can be called on TensorFlow tensors:
x = Dense(128, activation='relu')(img)  # fully-connected layer with 128 units and ReLU activation
x = Dense(128, activation='relu')(x)
preds = Dense(10, activation='softmax')(x)  # output layer with 10 units and a softmax activation

# Place-holder for the labels and loss function
labels = tf.placeholder(tf.float32, shape=(None, 10))

from keras.objectives import categorical_crossentropy
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# Model training with MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Run training loop
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1]})

# Run the model and print the accuracy
from keras.metrics import categorical_accuracy as accuracy

acc_value = accuracy(labels, preds)
with sess.as_default():
    print( "Accuracy = ",  acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels}) )
print("Done running model");

# Instrument the graph with TensorFI
fi = ti.TensorFI(sess, logLevel = 100)
fi.turnOnInjections();
with sess.as_default():
    print( "Accuracy = ",  acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels}) )
print("Done running instrumented model");
