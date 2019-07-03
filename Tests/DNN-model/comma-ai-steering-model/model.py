import tensorflow as tf
import scipy
'''
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')
'''


x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

x_image = x

keep_prob = 0.

#first convolutional layer
conv1 = tf.layers.conv2d(
    # size of the model
    #####   our claim is that Rambo does not provide tensorflow version. And it is the combination of nvidia model
    ####   and comma ai model. We implement both.
    x_image,
    filters=16,
    kernel_size=(8, 8),
    strides=(4, 4),
    padding='same',
    kernel_initializer=tf.contrib.layers.xavier_initializer(),
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    name='conv1',
    activation=tf.nn.elu
)


conv2 = tf.layers.conv2d(
    conv1,
    filters=32,
    kernel_size=(5, 5),
    strides=(2, 2),
    padding='same',
    kernel_initializer=tf.contrib.layers.xavier_initializer(),
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    name='conv2',
    activation=tf.nn.elu
)

conv3 = tf.layers.conv2d(
    conv2,
    filters=64,
    kernel_size=(5, 5),
    strides=(2, 2),
    padding='same',
    kernel_initializer=tf.contrib.layers.xavier_initializer(),
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    name='conv3',
    activation=None
)

#f1 = tf.contrib.layers.flatten(conv3)
f1 = tf.reshape(conv3, [-1, 4160])

#d1 = tf.nn.dropout(f1, keep_prob1=0.2)
e1 = tf.nn.elu(f1)
dense1 = tf.layers.dense(e1, units=512)
#d2 = tf.nn.dropout(dense1, keep_prob2=0.5)
e2 = tf.nn.elu(dense1)


y = tf.layers.dense(e2, units=1)


