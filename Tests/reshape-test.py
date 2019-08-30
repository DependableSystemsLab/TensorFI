#!/usr/bin/python

# Code example to deal with issue opened on TensorFI Github repo

from __future__ import print_function

import imp

# Override Python path to load TensorFI from local installation
(fileName, package, desc) = imp.find_module("tensorFI", ["/home/karthikp/Programs/TensorFI/TensorFI"])
imp.load_module( "TensorFI", fileName, package, desc )

import tensorflow as tf
import TensorFI as ti
import numpy as np

x = tf.placeholder(shape=(None, 2, 2), dtype='float32')

def my_func(arg,x):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    t_arg = tf.layers.flatten(x)
    return tf.matmul(t_arg, arg)

value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0],[2.1,2.5],[3.0,4.0]], dtype=np.float32),x)

s = tf.Session()
init = tf.global_variables_initializer()
print("Initial : ", s.run(init))

fi = ti.TensorFI(s,name = "var")
fi.turnOnInjections()
fi.setLogLevel(10)

logs_path = "./logs"
logWriter = tf.summary.FileWriter( logs_path, s.graph )

print("variable test : ", s.run(value_3,feed_dict = {x:[[[1.0,2.0],[3.0,4.0]]]}))
