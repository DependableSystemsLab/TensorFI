# These are the list of fault injection functions for different types of faults
# NOTE: There are separate versions of the scalar and tensor values for portability
# If you add a new fault type, please create both the scalar and tensor functions 

import numpy as np
from math import floor

# Currently, we support 6 types of faults {None, Rand, Zero, Rand-element, bitFlip-element, bitFlip-tensor} - See fiConfig.py

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

def randomElementScalar( dtype, max = 1.0):
	"Return a random value of type dtype from [0, max]"
	return dtype.type( np.random.random() * max )

def randomElementTensor ( dtype, val):
	"Random replacement of an element in a tensor with another one"
	"Only one element in a tensor will be changed while the other remains unchanged" 
	dim = val.ndim 
	
	if(dim==1):
		index = np.random.randint(low=0 , high=(val.shape[0]))
		val[index] = np.random.random() 
	elif(dim==2):
		index = [np.random.randint(low=0 , high=(val.shape[0])) , np.random.randint(low=0 , high=(val.shape[1]))]
		val[ index[0] ][ index[1] ] = np.random.random()

	return dtype.type( val )



def float2bin(number, decLength = 10): 
	"convert float data into binary expression"
	# we consider fixed-point data type, 32 bit: 1 sign bit, 21 integer and 10 mantissa

	# split integer and decimal part into seperate variables  
	integer, decimal = str(number).split(".") 
	# convert integer and decimal part into integer  
	integer = int(integer)  
	# Convert the integer part into binary form. 
	res = bin(integer)[2:] + "."		# strip the first binary label "0b"

	# 21 integer digit, 22 because of the decimal point "."
	res = res.zfill(22)
	
	def decimalConverter(decimal): 
		"E.g., it will return `x' as `0.x', for binary conversion"
		decimal = '0' + '.' + decimal 
		return float(decimal)

	# iterate times = length of binary decimal part
	for x in range(decLength): 
		# Multiply the decimal value by 2 and seperate the integer and decimal parts 
		# formating the digits so that it would not be expressed by scientific notation
		integer, decimal = format( (decimalConverter(decimal)) * 2, '.10f' ).split(".")    
		res += integer 

	return res 


def randomBitFlip(val):
	"Flip a random bit in the data to be injected"

	# Split the integer part and decimal part in binary expression
	def getBinary(number):
		# float point datatype
		if (isinstance(number, float)):
			binVal = float2bin(number)
			# split data into integer and decimal part
			integer, dec = binVal.split(".")
		# integer data type
		else:
			integer = bin(int(number)).lstrip("0b")
			# 21 digits for integer
			integer = integer.zfill(21)
			# To prevent extra digits in integer
			integer = integer[0:21]
			# integer has no mantissa
			dec = ''
		return integer, dec

	# we use a tag for the sign of negative val, and then consider all values as positive values
	# the sign bit will be tagged back when finishing bit flip
	negTag = 1
	if(str(val)[0]=="-"):
		negTag=-1

	if(isinstance(val, np.bool_)):	
		# boolean value
		return bool( (val+1)%2 )
	else:	
		# turn the val into positive val
		val = abs(val)
		integer, dec = getBinary(val)

	intLength = len(integer)
	decLength = len(dec)

	# random index of the bit to flip (switch to low=0 if you don't want to inject fault to sign bit)
	index = np.random.randint(low=-1 , high = intLength + decLength)
 
 	# injection index -1 is to flip the sign bit 

	if(index==-1):
		return val*negTag*(-1)

	# bit to flip at the integer part
	if(index < intLength):		
		# bit flipped from 1 to 0, thus minusing the corresponding value
		if(integer[index] == '1'):	val -= pow(2 , (intLength - index - 1))  
		# bit flipped from 0 to 1, thus adding the corresponding value
		else:						val += pow(2 , (intLength - index - 1))
	# bit to flip at the decimal part  
	else:						
		index = index - intLength 	  
		# bit flipped from 1 to 0, thus minusing the corresponding value
		if(dec[index] == '1'):	val -= 2 ** (-index-1)
		# bit flipped from 0 to 1, thus adding the corresponding value
		else:					val += 2 ** (-index-1) 

	return val*negTag

def bitElementScalar( dtype, val ):
	"Flip one bit of the scalar value"   
	return dtype.type( randomBitFlip(val) )

def bitElementTensor( dtype, val):
	"Flip ont bit of a random element in a tensor"
	# flatten the tensor into a vector and then restore the original shape in the end
	valShape = val.shape
	val = val.flatten()
	# select a random data item in the data space for injection
	index = np.random.randint(low=0, high=len(val))
	val[index] = randomBitFlip(val[index])	
	val = val.reshape(valShape)

	return dtype.type( val )

def bitScalar( dtype, val):
	"Flip one bit of the scalar value"
	return dtype.type( randomBitFlip(val) )

def bitTensor ( dtype, val):
	"Flip one bit in all elements within the tensor"
	# flatten the tensor into a vector and then restore the original shape in the end
	valShape = val.shape
	val = val.flatten()
	for i in range(len(val)):
		val[i] = randomBitFlip(val[i])
	val = val.reshape(valShape)
	return dtype.type( val )

