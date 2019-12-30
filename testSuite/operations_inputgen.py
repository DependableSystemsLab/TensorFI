#!/usr/bin/python

# This file holds all the functions that generate test inputs for the operations in operations_runTests.py
# The inputgenMap table at the end of the file maps each operation supported by TensorFI to one of the functions in this 

# To add support of an operation to this test script, add the operation to inputgenMap
# If a function exists that already supports the input requirements of the new operation you can map to that function; otherwise you must write a new function to specify the input test cases

# each inputgen function must return a list object containing each set of inputs for the test (i.e., a list of lists)
# the returned inputs list should be a list of all the sets of inputs you wish to test for a specific operation

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
    # tf.math.multiply( x, y, name=None )

    # x: A Tensor. Must be one of the following types: bfloat16, half, float32, float64, uint8, int8, uint16, int16, int32, int64, complex64, complex128.
    # y: A Tensor. Must have the same type as x.
    # name: A name for the operation (optional).

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

def inputgen_Square():
    # operations supported:
    # tf.math.square( x, name=None )
    # tf.shape( input, name=None, out_type=tf.dtypes.int32)
    # tf.math.negative( x, name=None )

    # x: A Tensor. Must be one of the following types: bfloat16, half, float32, float64, int32, int64, complex64, complex128.
    # name: A name for the operation (optional).
    # input: A Tensor or SparseTensor.
    # out_type: (Optional) The specified output type of the operation (int32 or int64). Defaults to tf.int32.

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
            random.seed(i+j)
            input_x = tf.constant(random.sample(rand_ints, num_elements), shape=(i,j), dtype=tf.int32)
            inputs.append([input_x])

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
            random.seed(i+j)
            input_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            inputs.append([input_x])

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
            random.seed(j)
            imag_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            input_x = tf.complex(real_x,imag_x)
            inputs.append([input_x])

    return inputs

def inputgen_Identity():
    # operations supported:
    # tf.identity( input, name=None )
    # tf.size( input, name=None, out_type=tf.dtypes.int32 )
    # tf.rank( input, name=None )

    # input: A Tensor. (any type)
    # name: A name for the operation (optional).
    # out_type: (Optional) The specified non-quantized numeric output type of the operation. Defaults to tf.int32.

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
            random.seed(i+j)
            input_x = tf.constant(random.sample(rand_ints, num_elements), shape=(i,j), dtype=tf.int32)
            inputs.append([input_x])

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
            random.seed(i+j)
            input_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            inputs.append([input_x])

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
            random.seed(j)
            imag_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            input_x = tf.complex(real_x,imag_x)
            inputs.append([input_x])
    
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
            random.seed(i+j)
            input_x = tf.constant(random.sample(rand_strings, num_elements), shape=(i,j))
            inputs.append([input_x])

    return inputs

def inputgen_Fill():
    # operations supported:
    # tf.fill( dims, value, name=None )

    # dims: A Tensor. Must be one of the following types: int32, int64. 1-D. Represents the shape of the output tensor.
    # value: A Tensor. 0-D (scalar). Value to fill the returned tensor.
    # name: A name for the operation (optional).

    # general approach: create tensors of varying shapes (both input shapes must match) filled with random constant numbers

    inputs = [] # each item in this list is a set of inputs passed to a create_op() in the main script

    # datatype: int
    rand_ints = []
    for x in range(0,100):
        random.seed(x)
        rand_ints.append(random.randint(-100,100))
    # create inputs
    for i in range(1,10):
        for j in range(1,10):
            # shape of tensor is (i,j)
            random.seed(i+j)
            dims  = tf.constant([i,j], dtype=tf.int32)
            value = tf.constant(random.sample(rand_ints, 1), shape=[], dtype=tf.int32)
            inputs.append([dims,value])

    # datatype: float
    rand_floats = []
    for x in range(0,100):
        random.seed(x)
        rand_floats.append(random.uniform(-100,100))
    # create inputs
    for i in range(1,10):
        for j in range(1,10):
            # shape of tensor is (i,j)
            random.seed(i+j)
            dims  = tf.constant([i,j], dtype=tf.int32)
            value = tf.constant(random.sample(rand_floats, 1), shape=[], dtype=tf.float32)
            inputs.append([dims,value])

    # datatype: complex
    rand_floats = []
    for x in range(0,100):
        random.seed(x)
        rand_floats.append(random.uniform(-100,100))
    # create inputs
    for i in range(1,10):
        for j in range(1,10):
            # shape of tensor is (i,j)
            dims  = tf.constant([i,j], dtype=tf.int32)

            random.seed(i)
            val_real = tf.constant(random.sample(rand_floats, 1), shape=[], dtype=tf.float32)

            random.seed(j)
            val_imag = tf.constant(random.sample(rand_floats, 1), shape=[], dtype=tf.float32)

            value = tf.complex(val_real,val_imag)
            
            inputs.append([dims,value])

    return inputs

def inputgen_FloorMod():
    # operations supported:
    # tf.math.floormod( x, y, name=None )

    # x: A Tensor. Must be one of the following types: int32, int64, bfloat16, half, float32, float64.
    # y: A Tensor. Must have the same type as x.
    # name: A name for the operation (optional).

    # general approach: create tensors of varying shapes (both input shapes must match) filled with random constant numbers

    inputs = [] # each item in this list is a set of inputs passed to a create_op() in the main script

    # datatype: int
    rand_ints = []
    for x in range(0,100):
        random.seed(x)
        rand_int = random.randint(-100,100)
        if rand_int == 0:
            rand_int = 1 # to avoid division by zero
        rand_ints.append(rand_int)
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
        rand_float = random.uniform(-100,100)
        if rand_float == 0.0:
            rand_float = 1.0 # avoid division by zero
        rand_floats.append(rand_float)
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

    return inputs

def inputgen_Range():
    # operations supported:
    # tf.range( start, limit, delta=1, dtype=None, name='range' )

    # start: A 0-D Tensor (scalar). Acts as first entry in the range if limit is not None; otherwise, acts as range limit and first entry defaults to 0.
    # limit: A 0-D Tensor (scalar). Upper limit of sequence, exclusive. If None, defaults to the value of start while the first entry of the range defaults to 0.
    # delta: A 0-D Tensor (scalar). Number that increments start. Defaults to 1.
    # dtype: The type of the elements of the resulting tensor.
    # name: A name for the operation. Defaults to "range".

    # general approach: create tensors of varying shapes (both input shapes must match) filled with random constant numbers

    inputs = [] # each item in this list is a set of inputs passed to a create_op() in the main script

    # datatype: int
    # create inputs
    for i in range(-100,100,5):
        for j in range(-100,100,5):
            if i > j:
                start = tf.constant(j, dtype=tf.int32)
                limit = tf.constant(i, dtype=tf.int32)
            elif j > i:
                start = tf.constant(i, dtype=tf.int32)
                limit = tf.constant(j, dtype=tf.int32)
            else:
                continue # i cannot equal j
            random.seed(i+j)
            delta = tf.constant( random.randint(1,10), dtype=tf.int32)
            inputs.append([start,limit,delta])

    # datatype: float
    # create inputs
    for i in range(-100,100,5):
        for j in range(-100,100,5):
            if i > j:
                start = tf.constant(j, dtype=tf.float32)
                limit = tf.constant(i, dtype=tf.float32)
            elif j > i:
                start = tf.constant(i, dtype=tf.float32)
                limit = tf.constant(j, dtype=tf.float32)
            else:
                continue # i cannot equal j
            random.seed(i+j)
            delta = tf.constant( random.uniform(1,10), dtype=tf.float32)
            inputs.append([start,limit,delta])

    return inputs

def inputgen_MatMul():
    # operations supported:
    # tf.linalg.matmul( a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None )

    # a: Tensor of type float16, float32, float64, int32, complex64, complex128 and rank > 1.
    # b: Tensor with same type and rank as a.
    # transpose_a: If True, a is transposed before multiplication.
    # transpose_b: If True, b is transposed before multiplication.
    # adjoint_a: If True, a is conjugated and transposed before multiplication.
    # adjoint_b: If True, b is conjugated and transposed before multiplication.
    # a_is_sparse: If True, a is treated as a sparse matrix.
    # b_is_sparse: If True, b is treated as a sparse matrix.
    # name: Name for the operation (optional).

    # shape of input a is (i,j)
    # shape of input b is (j,k)
    # result is a*b with shape (i,k)

    inputs = [] # each item in this list is a set of inputs passed to a create_op() in the main script

    # datatype: int
    rand_ints = []
    for x in range(0,1000):
        random.seed(x)
        rand_ints.append(random.randint(-1000,1000))
    for i in range(1,15):
        for j in range(1,15):
            for k in range(1,15):
                # generate input a
                num_elements_a = i * j
                random.seed(num_elements_a)
                input_a = tf.constant(random.sample(rand_ints, num_elements_a), shape=(i,j), dtype=tf.int32)
                # generate input b
                num_elements_b = j * k
                random.seed(num_elements_b)
                input_b = tf.constant(random.sample(rand_ints, num_elements_b), shape=(j,k), dtype=tf.int32)

                inputs.append([input_a,input_b])

    # datatype: float
    rand_floats = []
    for x in range(0,1000):
        random.seed(x)
        rand_floats.append(random.uniform(-1000,1000))
    for i in range(1,15):
        for j in range(1,15):
            for k in range(1,15):
                # generate input a
                num_elements_a = i * j
                random.seed(num_elements_a)
                input_a = tf.constant(random.sample(rand_floats, num_elements_a), shape=(i,j), dtype=tf.float32)
                # generate input b
                num_elements_b = j * k
                random.seed(num_elements_b)
                input_b = tf.constant(random.sample(rand_floats, num_elements_b), shape=(j,k), dtype=tf.float32)

                inputs.append([input_a,input_b])

    # datatype: complex
    rand_floats = []
    for x in range(0,1000):
        random.seed(x)
        rand_floats.append(random.uniform(-1000,1000))
    for i in range(1,15):
        for j in range(1,15):
            for k in range(1,15):
                # generate input a
                num_elements_a = i * j
                random.seed(num_elements_a)
                real_a = tf.constant(random.sample(rand_floats, num_elements_a), shape=(i,j), dtype=tf.float32)
                random.seed(num_elements_a + i)
                imag_a = tf.constant(random.sample(rand_floats, num_elements_a), shape=(i,j), dtype=tf.float32)
                input_a = tf.complex(real_a,imag_a)
                # generate input b
                num_elements_b = j * k
                random.seed(num_elements_b)
                real_b = tf.constant(random.sample(rand_floats, num_elements_b), shape=(j,k), dtype=tf.float32)
                random.seed(num_elements_b + k)
                imag_b = tf.constant(random.sample(rand_floats, num_elements_b), shape=(j,k), dtype=tf.float32)
                input_b = tf.complex(real_b,imag_b)

                inputs.append([input_a,input_b])

    return inputs

def inputgen_ArgMax():
    # operations supported:
    # tf.math.argmax( input, axis=None, name=None, dimension=None, output_type=tf.dtypes.int64 )
    # tf.math.argmin( input, axis=None, name=None, dimension=None, output_type=tf.dtypes.int64 )

    # input: A Tensor. Must be one of the following types: float32, float64, int32, uint8, int16, int8, complex64, int64, qint8, quint8, qint32, bfloat16, uint16, complex128, half, uint32, uint64.
    # axis: A Tensor. Must be one of the following types: int32, int64. int32 or int64, must be in the range [-rank(input), rank(input)). Describes which axis of the input Tensor to reduce across. For vectors, use axis = 0.
    # output_type: An optional tf.DType from: tf.int32, tf.int64. Defaults to tf.int64.
    # name: A name for the operation (optional).

    # general approach: create tensors of varying shapes (both input shapes must match) filled with random constant numbers

    inputs = [] # each item in this list is a set of inputs passed to a create_op() in the main script

    # datatype: int
    rand_ints = []
    for x in range(0,1000):
        random.seed(x)
        rand_ints.append(random.randint(-1000,1000))
    # create inputs of different tensor shapes
    for i in range(2,15):
        for j in range(2,15):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i+j)
            input_x = tf.constant(random.sample(rand_ints, num_elements), shape=(i,j), dtype=tf.int32)
            axis = tf.constant(random.randint(0,1), dtype=tf.int32)
            inputs.append([input_x, axis])

    # datatype: float
    rand_floats = []
    for x in range(0,1000):
        random.seed(x)
        rand_floats.append(random.uniform(-1000,1000))
    # create inputs of different tensor shapes
    for i in range(2,15):
        for j in range(2,15):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i+j)
            input_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            axis = tf.constant(random.randint(0,1), dtype=tf.int32)
            inputs.append([input_x, axis])

    # datatype: complex
    if not tf.test.is_gpu_available():
        # NOTE: must have a GPU to test the remaining inputs
        return inputs
    
    rand_floats = []
    for x in range(0,1000):
        random.seed(x)
        rand_floats.append(random.uniform(-1000,1000))
    # create inputs of different tensor shapes
    for i in range(2,10):
        for j in range(2,10):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i)
            real_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            random.seed(j)
            imag_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            input_x = tf.complex(real_x,imag_x)

            axis = tf.constant(random.randint(0,1))
            inputs.append([input_x, axis])

    return inputs

def inputgen_Equal():
    # operations supported:
    # tf.math.equal( x, y, name=None )
    # tf.math.not_equal( x, y, name=None )

    # x: A Tensor. Must be one of the following types: bfloat16, half, float32, float64, uint8, int8, int16, int32, int64, complex64, quint8, qint8, qint32, string, bool, complex128.
    # y: A Tensor. Must have the same type as x.
    # name: A name for the operation (optional).

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
            if random.choice([True,False]):
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
            if random.choice([True,False]):
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
            if random.choice([True,False]):
                random.seed(i)
                real_y = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
                random.seed(i*2)
                imag_y = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            else:
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
            if random.choice([True,False]):
                random.seed(j)
            input_y = tf.constant(random.sample(rand_strings, num_elements), shape=(i,j))
            inputs.append([input_x,input_y])

    # datatype: bool
    rand_bools = []
    for x in range(0,100):
        random.seed(x)
        rand_bools.append(random.choice([True,False]))
    # create inputs of different tensor shapes
    for i in range(1,10):
        for j in range(1,10):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i)
            input_x = tf.constant(random.sample(rand_bools, num_elements), shape=(i,j), dtype=tf.bool)
            if random.choice([True,False]):
                random.seed(j)
            input_y = tf.constant(random.sample(rand_bools, num_elements), shape=(i,j), dtype=tf.bool)
            inputs.append([input_x,input_y])

    return inputs

def inputgen_LessEqual():
    # operations supported:
    # tf.math.less_equal( x, y, name=None )

    # x: A Tensor. Must be one of the following types: float32, float64, int32, uint8, int16, int8, int64, bfloat16, uint16, half, uint32, uint64.
    # y: A Tensor. Must have the same type as x.
    # name: A name for the operation (optional).

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
            if random.choice([True,False]):
                input_y = tf.constant(100, shape=(i,j), dtype=tf.int32)
            else:
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
            if random.choice([True,False]):
                input_y = tf.constant(100.0, shape=(i,j), dtype=tf.float32)
            else:
                random.seed(j)
                input_y = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            inputs.append([input_x,input_y])

    return inputs

def inputgen_Cast():
    # operations supported:
    # tf.dtypes.cast( x, dtype, name=None )

    # x: A Tensor or SparseTensor or IndexedSlices of numeric type. It could be uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float32, float64, complex64, complex128, bfloat16.
    # dtype: The destination type. The list of supported dtypes is the same as x.
    # name: A name for the operation (optional).

    # general approach: create tensors of varying shapes (both input shapes must match) filled with random constant numbers

    inputs = [] # each item in this list is a set of inputs passed to a create_op() in the main script

    dtypes = [tf.uint8, tf.uint16, tf.uint32, tf.uint64, tf.int8, tf.int16, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64, tf.complex64, tf.complex128, tf.bfloat16]

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
            random.seed(i+j)
            input_x = tf.constant(random.sample(rand_ints, num_elements), shape=(i,j), dtype=tf.int32)
            dtype = random.choice(dtypes)
            inputs.append([input_x, dtype])

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
            random.seed(i+j)
            input_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            dtype = random.choice(dtypes)
            inputs.append([input_x, dtype])

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
            random.seed(j)
            imag_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            input_x = tf.complex(real_x,imag_x)

            dtype = random.choice(dtypes)
            inputs.append([input_x, dtype])

    return inputs

def inputgen_Mean():
    # operations supported:
    # tf.math.reduce_mean( input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None, keep_dims=None )

    # input_tensor: The tensor to reduce. Should have numeric type.
    # axis: The dimensions to reduce. If None (the default), reduces all dimensions. Must be in the range [-rank(input_tensor), rank(input_tensor)).
    # keepdims: If true, retains reduced dimensions with length 1.
    # name: A name for the operation (optional).
    # reduction_indices: The old (deprecated) name for axis.
    # keep_dims: Deprecated alias for keepdims.

    inputs = [] # each item in this list is a set of inputs passed to a create_op() in the main script

    # datatype: int
    rand_ints = []
    for x in range(0,1000):
        random.seed(x)
        rand_ints.append(random.randint(-1000,1000))
    # create inputs of different tensor shapes
    for i in range(2,15):
        for j in range(2,15):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i+j)
            input_x = tf.constant(random.sample(rand_ints, num_elements), shape=(i,j), dtype=tf.int32)
            axis = tf.constant(random.randint(0,1), dtype=tf.int32)
            inputs.append([input_x, axis])

    # datatype: float
    rand_floats = []
    for x in range(0,1000):
        random.seed(x)
        rand_floats.append(random.uniform(-1000,1000))
    # create inputs of different tensor shapes
    for i in range(2,15):
        for j in range(2,15):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i+j)
            input_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            axis = tf.constant(random.randint(0,1), dtype=tf.int32)
            inputs.append([input_x, axis])

    return inputs

def inputgen_NonZero():
    # operations supported:
    # tf.math.count_nonzero( input_tensor=None, axis=None, keepdims=None, dtype=tf.dtypes.int64, name=None, reduction_indices=None, keep_dims=None, input=None )

    # input_tensor: The tensor to reduce. Should be of numeric type, bool, or string.
    # axis: The dimensions to reduce. If None (the default), reduces all dimensions. Must be in the range [-rank(input_tensor), rank(input_tensor)).
    # keepdims: If true, retains reduced dimensions with length 1.
    # dtype: The output dtype; defaults to tf.int64.
    # name: A name for the operation (optional).
    # reduction_indices: The old (deprecated) name for axis.
    # keep_dims: Deprecated alias for keepdims.
    # input: Overrides input_tensor. For compatibility.

    inputs = [] # each item in this list is a set of inputs passed to a create_op() in the main script

    # datatype: int
    rand_ints = []
    for x in range(0,100):
        random.seed(x)
        if random.choice([True,False]):
            rand_ints.append(int(0))
        else:
            rand_ints.append(random.randint(-100,100))
    # create inputs of different tensor shapes
    for i in range(1,10):
        for j in range(1,10):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i+j)
            input_x = tf.constant(random.sample(rand_ints, num_elements), shape=(i,j), dtype=tf.int32)
            axis = tf.constant(random.randint(0,1), dtype=tf.int32)
            inputs.append([input_x, axis])

    # datatype: float
    rand_floats = []
    for x in range(0,100):
        random.seed(x)
        if random.choice([True,False]):
            rand_floats.append(float(0.0))
        else:
            rand_floats.append(random.uniform(-100.0,100.0))
    # create inputs of different tensor shapes
    for i in range(1,10):
        for j in range(1,10):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i+j)
            input_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            axis = tf.constant(random.randint(0,1), dtype=tf.int32)
            inputs.append([input_x, axis])

    # datatype: complex
    rand_floats = []
    for x in range(0,100):
        random.seed(x)
        if random.choice([True,False]):
            rand_floats.append(float(0.0))
        else:
            rand_floats.append(random.uniform(-100,100))
    # create inputs of different tensor shapes
    for i in range(1,10):
        for j in range(1,10):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i)
            real_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            random.seed(j)
            imag_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            input_x = tf.complex(real_x,imag_x)
            axis = tf.constant(random.randint(0,1), dtype=tf.int32)
            inputs.append([input_x, axis])

    # datatype: string
    rand_strings = []
    N = 8 # size of random string
    for x in range(0,100):
        random.seed(x)
        if random.choice([True,False]):
            rand_strings.append('')
        else:
            rand_strings.append(''.join(random.choice(string.ascii_letters + string.punctuation) for x in range(N)))
    # create inputs of different tensor shapes
    for i in range(1,10):
        for j in range(1,10):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i)
            input_x = tf.constant(random.sample(rand_strings, num_elements), shape=(i,j))
            axis = tf.constant(random.randint(0,1), dtype=tf.int32)
            inputs.append([input_x,axis])

    return inputs

def inputgen_Reshape():
    # operations supported:
    # tf.reshape( tensor, shape, name=None )

    # tensor: A Tensor.
    # shape: A Tensor. Must be one of the following types: int32, int64. Defines the shape of the output tensor.
    # name: A name for the operation (optional).

    inputs = [] # each item in this list is a set of inputs passed to a create_op() in the main script

    # datatype: int
    rand_ints = []
    for x in range(0,1000):
        random.seed(x)
        rand_ints.append(random.randint(-100,100))
    # create inputs of different tensor shapes
    for i in range(2,15):
        for j in range(2,15):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i)
            input_x = tf.constant(random.sample(rand_ints, num_elements), shape=(i,j), dtype=tf.int32)
            for k in reversed(range(1,20)):
                if num_elements % k == 0:
                    shape = tf.constant([k,int(num_elements / k)],dtype=tf.int32)
                    if (k,int(num_elements/k)) != (i,j):
                        break
            inputs.append([input_x,shape])

    # datatype: float
    rand_floats = []
    for x in range(0,1000):
        random.seed(x)
        rand_floats.append(random.uniform(-100,100))
    # create inputs of different tensor shapes
    for i in range(2,15):
        for j in range(2,15):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i)
            input_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            for k in reversed(range(1,20)):
                if num_elements % k == 0:
                    shape = tf.constant([k,int(num_elements / k)],dtype=tf.int32)
                    if (k,int(num_elements/k)) != (i,j):
                        break
            inputs.append([input_x,shape])

    # datatype: string
    rand_strings = []
    for x in range(0,1000):
        random.seed(x)
        N = 8 # size of random string
        rand_strings.append(''.join(random.choice(string.ascii_letters + string.punctuation) for x in range(N)))
    # create inputs of different tensor shapes
    for i in range(2,15):
        for j in range(2,15):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i)
            input_x = tf.constant(random.sample(rand_strings, num_elements), shape=(i,j))
            for k in reversed(range(1,20)):
                if num_elements % k == 0:
                    shape = tf.constant([k,int(num_elements / k)],dtype=tf.int32)
                    if (k,int(num_elements/k)) != (i,j):
                        break
            inputs.append([input_x,shape])

    return inputs

def inputgen_Max():
    # operations supported:
    # tf.math.maximum( x, y, name=None )
    # tf.math.minimum( x, y, name=None )
    # tf.math.greater( x, y, name=None )

    # x: A Tensor. Must be one of the following types: bfloat16, half, float32, float64, int32, int64.
    # y: A Tensor. Must have the same type as x.
    # name: A name for the operation (optional).

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

    return inputs

def inputgen_Switch():
    # operations supported:
    # tf.keras.backend.switch( condition, then_expression, else_expression )

    # condition: tensor (int or bool).
    # then_expression: either a tensor, or a callable that returns a tensor.
    # else_expression: either a tensor, or a callable that returns a tensor.

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
            condition = tf.constant(random.choice([True,False]), dtype=tf.bool)
            inputs.append([condition,input_x,input_y])

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
            condition = tf.constant(random.choice([True,False]), dtype=tf.bool)
            inputs.append([condition,input_x,input_y])

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
            condition = tf.constant(random.choice([True,False]), dtype=tf.bool)
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
            condition = tf.constant(random.choice([True,False]), dtype=tf.bool)
            inputs.append([condition,input_x,input_y])

    return inputs

def inputgen_Pow():
    # operations supported:
    # tf.math.pow( x, y, name=None )

    # x: A Tensor of type float16, float32, float64, int32, int64, complex64, or complex128.
    # y: A Tensor of type float16, float32, float64, int32, int64, complex64, or complex128.
    # name: A name for the operation (optional).

    inputs = [] # each item in this list is a set of inputs passed to a create_op() in the main script

    # datatype: int
    rand_ints = []
    rand_ints_pos = []
    for x in range(0,100):
        random.seed(x)
        rand_ints.append(random.randint(-100,100))
        rand_ints_pos.append(random.randint(0,20))
    # create inputs of different tensor shapes
    for i in range(1,10):
        for j in range(1,10):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i)
            input_x = tf.constant(random.sample(rand_ints, num_elements), shape=(i,j), dtype=tf.int32)
            random.seed(j)
            input_y = tf.constant(random.sample(rand_ints_pos, num_elements), shape=(i,j), dtype=tf.int32)
            inputs.append([input_x,input_y])

    return inputs # return before float inputs (instrumented output is different from original with float inputs)

    # datatype: float
    rand_floats = []
    rand_floats_pos = []
    for x in range(0,100):
        random.seed(x)
        rand_floats.append(random.uniform(-50,50))
        rand_floats_pos.append(random.uniform(0,20))
    # create inputs of different tensor shapes
    for i in range(1,10):
        for j in range(1,10):
            # shape of tensor is (i,j)
            num_elements = i * j
            random.seed(i)
            input_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            random.seed(j)
            input_y = tf.constant(random.sample(rand_floats_pos, num_elements), shape=(i,j), dtype=tf.float32)
            inputs.append([input_x,input_y])

    # datatype: complex
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
            real_y = tf.constant(random.sample(rand_floats_pos, num_elements), shape=(i,j), dtype=tf.float32)
            random.seed(j*2)
            imag_y = tf.constant(random.sample(rand_floats_pos, num_elements), shape=(i,j), dtype=tf.float32)
            input_y = tf.complex(real_y,imag_y)
            inputs.append([input_x,input_y])

def inputgen_RealDiv():
    # operations supported:
    # tf.realdiv( x, y, name=None )

    # x: A Tensor. Must be one of the following types: bfloat16, half, float32, float64, uint8, int8, uint16, int16, int32, int64, complex64, complex128.
    # y: A Tensor. Must have the same type as x.
    # name: A name for the operation (optional).

    inputs = [] # each item in this list is a set of inputs passed to a create_op() in the main script

    # datatype: float
    rand_floats = []
    for x in range(0,100):
        random.seed(x)
        rand_float = random.uniform(-100,100)
        while rand_float == 0.0: # avoid divide by zero
            rand_float = random.uniform(-100,100)
        rand_floats.append(rand_float)
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
    
    return inputs # NOTE: skips the integer inputs because they throw an error

    # datatype: int
    rand_ints = []
    for x in range(0,100):
        random.seed(x)
        rand_int = random.randint(-100,100)
        while rand_int == 0: # avoid divide by zero
            rand_int = random.randint(-100,100)
        rand_ints.append(rand_int)
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

def inputgen_Abs():
    # operations supported:
    # tf.math.abs( x, name=None )

    # x: A Tensor or SparseTensor of type float16, float32, float64, int32, int64, complex64 or complex128.
    # name: A name for the operation (optional).

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
            random.seed(i+j)
            input_x = tf.constant(random.sample(rand_ints, num_elements), shape=(i,j), dtype=tf.int32)
            inputs.append([input_x])

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
            random.seed(i+j)
            input_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            inputs.append([input_x])

    return inputs

def inputgen_Tanh():
    # operations supported:
    # tf.math.tanh( x, name=None )

    # x: A Tensor. Must be one of the following types: bfloat16, half, float32, float64, complex64, complex128.
    # name: A name for the operation (optional).

    inputs = [] 

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
            random.seed(i+j)
            input_x = tf.constant(random.sample(rand_floats, num_elements), shape=(i,j), dtype=tf.float32)
            inputs.append([input_x])

    return inputs

# This table is used to store all of the operations that will be tested by operations_runTests.py
# By default this should contain all the operations currently supported by TensorFI
# When implementing a new operation in TensorFI, add an entry to the list below for that operation

# Each dictionary entry is in the following form
#       op_type: inputgen_function

# op_type: (String) The op_type that is passable to the tf.create_op(op_type, inputs) function, i.e., the "type" property of the Operation object (op.type) and should be same as the opTable entry in injectFault.py
# inputgen_function: (Function) The function that is called to generate the set of test inputs. Depends on the types of inputs the operation supports (refer to the tensorflow documentation for each operation). Try to re-use functions for other operations if they fit.

inputgenMap = {
    "Identity": inputgen_Identity,
    "Add": inputgen_Add,
    "Sub": inputgen_Sub,
    "Mul": inputgen_Sub,
    "Square": inputgen_Square,
    "Shape": inputgen_Square,
    "Size": inputgen_Identity,
    "Fill": inputgen_Fill,
    "FloorMod": inputgen_FloorMod,
    "Range": inputgen_Range,
    "Rank": inputgen_Identity,
    "MatMul": inputgen_MatMul,
    "ArgMax": inputgen_ArgMax,
    "ArgMin": inputgen_ArgMax,
    "Equal": inputgen_Equal,
    "NotEqual": inputgen_Equal,
    "LessEqual": inputgen_LessEqual,
    "Mean": inputgen_Mean,
    "Reshape": inputgen_Reshape,
    "Maximum": inputgen_Max,
    "Minimum": inputgen_Max,
    "Greater": inputgen_Max,
    "Neg": inputgen_Square,
    "RealDiv": inputgen_RealDiv,
    "Abs": inputgen_Abs,
    "Tanh": inputgen_Tanh,
    #"Assign": ,
    #"Rsqrt": ,
    #"Log": ,
    #"Conv2D": ,
    #"Relu": ,
    #"MaxPool": ,
    #"Softmax": ,
    #"ExpandDims": ,
    #"BiasAdd": ,
    #"Sigmoid": ,
    #"Pack": ,
    #"Sum": ,
    #"Unpack": ,
    #"Pow": inputgen_Pow, # NOTE: operation fails when using input tensors of type float (instrumented graph outputs are different)
    #"Count_nonzero": inputgen_NonZero, # NOTE: this returns an error, seems like "Count_nonzero" is not a valid op name for tf.math.count_nonzero. Need to look into this
    #"Switch": inputgen_Switch, # NOTE: returns an error. We assume "Switch" refers to tf.keras.backend.switch but that may be incorrect
    #"Cast": inputgen_Cast, # NOTE: this raises an exception, apparently cannot pass the dtype parameter to create_op(), must figure out a way around this
    "end_of_ops": None # placeholder for end of list
}
