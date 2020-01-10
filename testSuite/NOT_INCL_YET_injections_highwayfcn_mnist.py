import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time

import TensorFI as ti

mnist_data = input_data.read_data_sets("MNIST_data", one_hot=True)

def weights_init(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def bias_init(shape, bias_init=0.05):
    return tf.Variable(tf.constant(bias_init, shape=shape))

def fully_connected_layer(input, input_shape, output_shape, activation=tf.nn.relu):
    weights = weights_init([input_shape, output_shape])
    bias = bias_init([output_shape])
    layer = tf.add(tf.matmul(input, weights), bias) #x*w + b
    
    if activation != None:
        return activation(layer)
    else:
        return layer

def highway_fc_layer(input, hidden_layer_size, carry_b = -2.0, activation=tf.nn.relu):
    #Step 1. Define weights and biases for the activation gate
    weights_normal = weights_init([hidden_layer_size, hidden_layer_size])
    bias_normal = bias_init([hidden_layer_size])
    
    #Step 2. Define weights and biases for the transform gate
    weights_transform = weights_init([hidden_layer_size, hidden_layer_size])
    bias_transform = bias_init(shape=[hidden_layer_size], bias_init=carry_b)
    
    #Step 3. calculate activation gate
    H = activation(tf.matmul(input, weights_normal) + bias_normal, name="Input_gate")
    #Step 4. calculate transform game
    T = tf.nn.sigmoid(tf.matmul(input, weights_transform) +bias_transform, name="T_gate")
    #Step 5. calculate carry get (1 - T)
    C = tf.subtract(1.0, T, name='C_gate')
    # y = (H * T) + (x * C)
    #Final step 6. campute the output from the highway fully connected layer
    y = tf.add(tf.multiply(H, T), tf.multiply(input, C), name='output_highway')
    return y

#defining hyperparams
input_shape = 784 #28x28x1 <- Number of pixels of MNIST image

hidden_size = 50 # This is number of neurons used at EVERY hidden highway layer, you can test with this number
                #but becuase we have highway (deep) network this number doesn't have to be very large

output_size = 10 # number of neurons at the output layer, 10 because we have 10 classes

number_of_layers = 18 # this is another hyperparam to care about in highway networks, play with it 

cary_bias = -20.0 # This is cary bias used at transform gate inside highway layer

epochs = 40 # How many times are we going to run through whole dataset

batch_size = 64 # How many data samples to feed to a network at onces

learning_rate = 0.01

#Defining inputs to tensorflow graph, one is for images - inputs, and another one is for classes - targets
inputs = tf.placeholder(tf.float32, shape=[None, input_shape], name='Input')
targets = tf.placeholder(tf.float32, shape=[None, output_size], name='output')

#Defining HIGHWAY NETWORK
prev_layer = None
output = None
for layer in range(number_of_layers):
    
    if layer == 0:
        #This for input layer
        prev_layer = fully_connected_layer(inputs, input_shape, hidden_size)
    elif layer == number_of_layers-1:
        #This if for output layer
        output = fully_connected_layer(prev_layer, hidden_size, output_size, activation=None)
    else:
        # for any layer between input and output layer
        prev_layer = highway_fc_layer(prev_layer, hidden_size, carry_b=cary_bias)

#Defining error/cost/loss function and optimizier
cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=targets)) #this is standard cross entropy loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#This is used only for testing
y_pred = tf.nn.softmax(output)
y_pred_scores = tf.argmax(y_pred, 1)
y_true = tf.argmax(targets, 1)

#Getting accuracy
correct_prediction = tf.equal(y_pred_scores, y_true)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

# if you make some mistake or change the structure of your network, good practice is to reset default graph.
# tf.reset_default_graph()

session = tf.Session()

session.run(tf.global_variables_initializer())

def optimize():
    
    for i in range(epochs):
        epoch_cost = []
        epoch_time = time.time()
        for ii in range(mnist_data.train.num_examples//batch_size):
            batch = mnist_data.train.next_batch(batch_size)
            imgs = batch[0]
            labs = batch[1]
            
            c, _ = session.run([cost, optimizer], feed_dict={inputs:imgs, targets:labs})

            epoch_cost.append(c)
        print("Epoch: {}/{}".format(i+1, epochs), " | Current loss: {}".format(np.mean(epoch_cost)),
             "  |  Epoch time: {:.2f}s".format(time.time() - epoch_time))
        print("test accuracy %g" % session.run(accuracy ,feed_dict={ inputs: mnist_data.test.images, targets: mnist_data.test.labels }))
    saver.save(session, './fcn')

def test_model():
    saver.restore(session, tf.train.latest_checkpoint('.'))
    return session.run(accuracy, feed_dict={inputs:mnist_data.test.images[:256], 
                                           targets:mnist_data.test.labels[:256]})

optimize()

print ("Accuracy is: ", test_model())

fi = ti.TensorFI(session, logLevel = 100, name = "fcn", disableInjections=False)

print ("Accuracy is: ", test_model())