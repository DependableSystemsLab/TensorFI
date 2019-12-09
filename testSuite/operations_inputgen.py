#!/usr/bin/python

# This file holds all the functions that generate test inputs for the operations in operations_runTests.py
# The inputgenMap table at the end of the file maps each operation supported by TensorFI to one of the functions in this 

# To add support of an operation to this test script, add the operation to inputgenMap
# If a function exists that already supports the input requirements of the new operation you can map to that function; otherwise you must write a new function to specify the input test cases

# each inputgen function must return a list object containing each set of inputs for the test (i.e., a list of lists)

# NOTE: for reproducibility the random numbers should use random.seed() to produce the same random numbers each time

import tensorflow as tf
import random
import string

def inputgen_simple():
    # basic inputgen function for example
    # generates list of input tensor pairs of various shapes
    # datatype: int
    inputs = []
    rand_ints = []
    for x in range(0,100):
        random.seed(x)
        rand_ints.append(random.randint(-100,100))
    # create inputs of different tensor shapes
    for i in range(1,10):
        for j in range(1,10):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i)
            input_x = tf.constant(random.sample(rand_ints, num_elements), shape=(i,j), dtype=tf.int32)
            random.seed(j)
            input_y = tf.constant(random.sample(rand_ints, num_elements), shape=(i,j), dtype=tf.int32)
            inputs.append([input_x,input_y])
    
    return inputs

def inputgen_Add():
    # operations supported:
    # tf.math.add( x, y, name=None )

    # x: A Tensor. Must be one of the following types: bfloat16, half, float32, float64, uint8, int8, int16, int32, int64, complex64, complex128, string.
    # y: A Tensor. Must have the same type as x.
    # name: A name for the operation (optional).

    # general approach: create tensors of varying shapes (both input shapes must match) filled with random constant numbers

    inputs = [] # each item in this list is a set of inputs passed to a create_op() in the main script

    # datatype: int
    rand_ints = []
    for x in range(0,100):
        random.seed(x)
        rand_ints.append(random.randint(-100,100))
    # create inputs of different tensor shapes
    for i in range(1,10):
        for j in range(1,10):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i)
            input_x = tf.constant(random.sample(rand_ints, num_elements), shape=(i,j), dtype=tf.int32)
            random.seed(j)
            input_y = tf.constant(random.sample(rand_ints, num_elements), shape=(i,j), dtype=tf.int32)
            inputs.append([input_x,input_y])

    # datatype: float
    rand_floats = []
    for x in range(0,100):
        random.seed(x)
        rand_floats.append(random.uniform(-100,100))
    # create inputs of different tensor shapes
    for i in range(1,10):
        for j in range(1,10):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i)
            input_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            random.seed(j)
            input_y = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            inputs.append([input_x,input_y])

    # datatype: complex
    rand_floats = []
    for x in range(0,100):
        random.seed(x)
        rand_floats.append(random.uniform(-100,100))
    # create inputs of different tensor shapes
    for i in range(1,10):
        for j in range(1,10):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i)
            real_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            random.seed(i*2)
            imag_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            input_x = tf.complex(real_x,imag_x)
            random.seed(j)
            real_y = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            random.seed(j*2)
            imag_y = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            input_y = tf.complex(real_y,imag_y)
            inputs.append([input_x,input_y])

    # datatype: string
    rand_strings = []
    for x in range(0,100):
        random.seed(x)
        N = 8 # size of random string
        rand_strings.append(''.join(random.choice(string.ascii_letters + string.punctuation) for x in range(N)))
    # create inputs of different tensor shapes
    for i in range(1,10):
        for j in range(1,10):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i)
            input_x = tf.constant(random.sample(rand_strings, num_elements), shape=(i,j))
            random.seed(j)
            input_y = tf.constant(random.sample(rand_strings, num_elements), shape=(i,j))
            inputs.append([input_x,input_y])

    return inputs

def inputgen_Sub():
    # operations supported:
    # tf.math.subtract( x, y, name=None )

    # x: A Tensor. Must be one of the following types: bfloat16, half, float32, float64, uint8, int8, uint16, int16, int32, int64, complex64, complex128.
    # y: A Tensor. Must have the same type as x.
    # name: A name for the operation (optional).

    # general approach: create tensors of varying shapes (both input shapes must match) filled with random constant numbers

    inputs = [] # each item in this list is a set of inputs passed to a create_op() in the main script

    # datatype: int
    rand_ints = []
    for x in range(0,100):
        random.seed(x)
        rand_ints.append(random.randint(-100,100))
    # create inputs of different tensor shapes
    for i in range(1,10):
        for j in range(1,10):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i)
            input_x = tf.constant(random.sample(rand_ints, num_elements), shape=(i,j), dtype=tf.int32)
            random.seed(j)
            input_y = tf.constant(random.sample(rand_ints, num_elements), shape=(i,j), dtype=tf.int32)
            inputs.append([input_x,input_y])

    # datatype: float
    rand_floats = []
    for x in range(0,100):
        random.seed(x)
        rand_floats.append(random.uniform(-100,100))
    # create inputs of different tensor shapes
    for i in range(1,10):
        for j in range(1,10):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i)
            input_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            random.seed(j)
            input_y = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            inputs.append([input_x,input_y])

    # datatype: complex
    rand_floats = []
    for x in range(0,100):
        random.seed(x)
        rand_floats.append(random.uniform(-100,100))
    # create inputs of different tensor shapes
    for i in range(1,10):
        for j in range(1,10):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i)
            real_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            random.seed(i*2)
            imag_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            input_x = tf.complex(real_x,imag_x)
            random.seed(j)
            real_y = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            random.seed(j*2)
            imag_y = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            input_y = tf.complex(real_y,imag_y)
            inputs.append([input_x,input_y])

    return inputs

# This table is used to store all of the operations that will be tested by operations_runTests.py
# By default this should contain all the operations currently supported by TensorFI
# When implementing a new operation in TensorFI, add an entry to the list below for that operation

# Each dictionary entry is in the following form
#       op_type: inputgen_function

# op_type: (String) The op_type that is passable to the tf.create_op(op_type, inputs) function, i.e., the "type" property of the Operation object (op.type) and should be same as the opTable entry in injectFault.py
# inputgen_function: (Function) The function that is called to generate the set of test inputs. Depends on the types of inputs the operation supports (refer to the tensorflow documentation for each operation). Try to re-use functions for other operations if they fit.

inputgenMap = {
    #"Assign": ,
    #"Identity": ,
    "Add": inputgen_Add,
    "Sub": inputgen_Sub,
    #"Mul": ,
    #"Square": ,
    #"Shape": ,
    #"Size": ,
    #"Fill": ,
    #"FloorMod": ,
    #"Range": ,
    #"Rank": ,
    #"Sum": ,
    #"MatMul": ,
    #"ArgMax": ,
    #"ArgMin": ,
    #"Equal": ,
    #"NotEqual": ,
    #"LessEqual": ,
    #"Cast": ,
    #"Mean": ,
    #"Count_nonzero": ,
    #"Reshape": ,
    #"Conv2D": ,
    #"Relu": ,
    #"MaxPool": ,
    #"Softmax": ,
    #"Maximum": ,
    #"Minimum": ,
    #"ExpandDims": ,
    #"Switch": ,
    #"Greater": ,
    #"Neg": ,
    #"Pow": ,
    #"RealDiv": ,
    #"Abs": ,
    #"Rsqrt": ,
    #"Log": ,
    #"BiasAdd": ,
    #"Sigmoid": ,
    #"Tanh": ,
    #"Pack": ,
    #"Unpack": ,
    "end_of_ops": inputgen_simple # placeholder for end of list
}
