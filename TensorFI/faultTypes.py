# These are the list of fault injection functions for different types of faults
# NOTE: There are separate versions of the scalar and tensor values for portability
# If you add a new fault type, please create both the scalar and tensor functions 

import numpy as np

# Currently, we support three types of faults { None, Rand, Zero } - See fiConfig.py

def randomScalar( dtype, max = 1.0 ):
	"Return a random value of type dtype from [0, max]"
	return dtype.type( np.random.random() * max )

def randomTensor( dtype, tensor):
	"Random replacement of a tensor value with another one"
	# The tensor.shape is a tuple, while rand needs linear arguments
	# So we need to unpack the tensor.shape tuples as arguments using *
	res = np.random.rand( *tensor.shape ) 
	return dtype.type( res )

def zeroScalar(dtype, val):
	"Return a scalar 0 of type dtype"
	# val is a dummy parameter for compatibility with randomScalar
	return dtype.type( 0.0 )

def zeroTensor(dtype, tensor):
	"Take a tensor and zero it"
	res = np.zeros( tensor.shape ) 
	return dtype.type( res )

def noScalar(dtype, val):
	"Dummy injection function that does nothing"
	return val

def noTensor(dtype, tensor):
	"Dummy injection function that does nothing"
	return tensor

