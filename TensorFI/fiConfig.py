# Fault injection configuration information: this is used for the global fault injector
from enum import Enum
import numpy as np
from faultTypes import *
import yaml
import logging

# These are the list of supported Operations below (if you add a new Op, please add it here)
class Ops(Enum):
	NOOP = "NOOP"
	ASSIGN = "ASSIGN"
	IDENTITY = "IDENTITY"
	ADD = "ADD"
	SUB = "SUB"
	MUL = "MUL"
	SQUARE = "SQUARE"
	SHAPE = "SHAPE"
	SIZE = "SIZE"
	FILL = "FILL"
	FLOOR = "FLOOR"
	FLOORMOD = "FLOOR-MOD"
	RANGE = "RANGE"
	RANK = "RANK"
	SUM = "SUM"
	MATMUL = "MATMUL"
	ARGMAX = "ARGMAX"
	ARGMIN = "ARGMIN"
	EQUAL = "EQUAL"
	NOT_EQUAL = "NOT-EQUAL"
	LESS_EQUAL = "LESS-EQUAL"
	GREATER_EQUAL = "GREATER-EQUAL"
	CAST = "CAST"
	MEAN = "MEAN"
	COUNT_NONZERO = "COUNT-NONZERO"
	RESHAPE = "RESHAPE"
	ONE_HOT = "ONE-HOT"
	CONV2D = "CONV2D"
	RELU = "RELU"
	MAXPOOL = "MAX-POOL"
	STRIDEDSLICE = "STRIDED-SLICE"
	SOFTMAX = "SOFT-MAX"
	MAXIMUM = "MAXIMUM"
	MINIMUM = "MINIMUM"
	EXPANDDIMS = "EXPAND-DIMS"
	SWITCH = "SWITCH"
	GREATER = "GREATER"
	NEGATIVE = "NEGATIVE"
	POWER = "POW"
	REALDIV = "REALDIV"
	ABSOLUTE = "ABSOLUTE"
	RSQRT = "RSQRT"
	LOG = "LOG"
	BIASADD = "BIASADD"
	SIGMOID = "SIGMOID"
	TANH = "TANH"
	PACK = "PACK"
	UNPACK = "UNPACK"
	ALL = "ALL"	# Chooses all the operations for injection (end of list)
	END = "END"  # Dummy operation for end of list
	LRN = "LRN" 
	ELU = "ELU"
	RANDOM_UNIFORM = "RANDOM-UNIFORM"
# End of Ops

# These are the list of supported Fault types below (if you add a new type, please add it here)
class FaultTypes(Enum):
	NONE = "None"
	RAND = "Rand"
	ZERO = "Zero"
	ELEM = "Rand-element"
	ELEMbit = "bitFlip-element"
	RANDbit = "bitFlip-tensor" 
# End of FaultTypes

# These are the list of supported Fields below (if you add a new Field, please add it here)
class Fields(Enum):
	ScalarFaultType = "ScalarFaultType"
	TensorFaultType = "TensorFaultType"
	Ops = "Ops"
	Seed = "Seed"
	SkipCount = "SkipCount"
	Instances = "Instances"
	InjectMode = "InjectMode"
# End of Fields

# These are the fault configuration functions
# The global class fiConf holds the config functions
class FIConfig(object):
	"Class to store configuration information about faults"

	# Static variable: Mapping from fault types to fault injection functions
	faultTypeMap = { 
		FaultTypes.NONE.value : (noScalar, noTensor),
		FaultTypes.RAND.value : (randomScalar, randomTensor),
		FaultTypes.ZERO.value : (zeroScalar, zeroTensor),
		FaultTypes.ELEM.value : (randomElementScalar, randomElementTensor),
		FaultTypes.ELEMbit.value : (bitElementScalar, bitElementTensor),
		FaultTypes.RANDbit.value : (bitScalar, bitTensor)
	}

	def faultConfigType(self, faultTypeScalar, faultTypeTensor):
		"Configure the fault injection type for Scalars and Tensors"

		# Check if the fault type is known and if so, assign the scalar functions 
		if self.faultTypeMap.has_key(faultTypeScalar):
			self.faultTypeScalar = faultTypeScalar
			self.injectScalar = self.faultTypeMap[ faultTypeScalar ][0]
		else:
			# If it's not known, declare an error
			raise ValueError("Unknown fault type " + str(faultTypeScalar))

		# Check if the fault type is known and if so, assign the tensor functions  
		if self.faultTypeMap.has_key(faultTypeTensor):
			self.faultTypeTensor = faultTypeTensor
			self.injectTensor = self.faultTypeMap[ faultTypeTensor ][1]  
		else:
			# If it's not known, declare an error
			raise ValueError("Unknown fault type " + str(faultTypeTensor))

	def faultConfigOp(self, opType, prob = 1.0):
		"Configure the fault injection operations"
		# Check if it's a defined operation, and if so, add it to the injectMap
		for op in Ops:
			if op.value==opType:	
				# Convert the prob to a 32 bit floating point before adding it
				probFP = np.float32(prob)

				# Check if the probability is a sane value
				if (probFP > 1.0 or probFP < 0.0):
					raise ValueError("Probability has to be in range [0,1]")

				# Finally, add the operation to the injectMap
				self.injectMap[ op ] = probFP

	def instanceConfigOp(self, opType, instance):
		"Configure the instance of each operations"
		# Check if it's a defined operation, and if so, add it to the opInstance
		for op in Ops:
			if op.value == opType:
				# Check if the instance is a sane value
				if (instance <= 0):
					raise ValueError("Instance has to be larger than 0")

				# Finally, add the operation to the injectMap
				self.opInstance[ op ] = int(instance)


	def isSelected(self, op):
		"Check if the op is among those selected for injection"
		# Either all operations are selected or this particular one is selected
		#	FIXME: Add specific operation categories here in the future
		return self.injectMap.has_key(op) or self.injectMap.has_key(Ops.ALL) 

	def getProbability(self, op):
		"Retreive the probability of the op for injection if it's present, otherwise return ALL"
		# Precondition: injectMap.has_key(op) or injectMap.has_key(Ops.ALL)
		if self.injectMap.has_key(op):
			return self.injectMap[ op ]
		else:
			return self.injectMap[ Ops.ALL ]

 	def getInstance (self, op):
 		"Retreive the instance of the op for injection if it's present"
		return self.opInstance[ op ] 			


	def __str__(self):
		"Convert this object to a string representation for printing"
		res = [ "FIConfig: {" ]
		res.append("\tfaultTypeScalar : " + str(self.faultTypeScalar) )
		res.append("\tfaultTypeTensor : " + str(self.faultTypeTensor) )
		res.append("\tinjectMap : "  + str(self.injectMap) )
		res.append("\tfaultSeed : " + str(self.faultSeed) )
		res.append("\tskipCount : " + str(self.skipCount) )
		res.append(" }")
		return "\n".join(res)

	def __init__(self,fiParams):
		"Configure the initial fault injection parameters from the fiParams Dictionary"
		# First configure the Scalar fault type
		# Default value of fault is NoFault
		if fiParams.has_key(Fields.ScalarFaultType.value):
			faultTypeScalar = fiParams[Fields.ScalarFaultType.value]
		else:	
			faultTypeScalar = "None"	
		# Next configure the Tensor fault type
		# Default value of fault is NoFault
		if fiParams.has_key(Fields.TensorFaultType.value):
			faultTypeTensor = str(fiParams[Fields.TensorFaultType.value])
		else:
			faultTypeTensor = "None"
		
		self.injectMode = ""
		if fiParams.has_key(Fields.InjectMode.value):
			self.injectMode = str(fiParams[Fields.InjectMode.value]) 
		else:
			# in this case, there will be no injection
			self.injectMode = "None"

		# Finally, call the faultConfigtype function with the parameters	
		self.faultConfigType(faultTypeScalar, faultTypeTensor)
	
		# Configure the operations to be included for instrumenting
		# default value is inject nothing (empty op list)
		self.injectMap = { }

		if fiParams.has_key(Fields.Ops.value):
			opsList = fiParams[Fields.Ops.value]
			if not opsList==None:
				for element in opsList:
					(opType, prob) = element.split('=')
					self.faultConfigOp(opType.rstrip(), prob.lstrip())

		# Configure the instances of each operation 
		self.opInstance = { }
		# Configure the amount of total instance of the algorithm, i.e., number of operations in the model
		self.totalInstance = 0

		if fiParams.has_key(Fields.Instances.value):
			instanceList = fiParams[Fields.Instances.value]
			if not instanceList==None:
				for element in instanceList:
					(opType, instance) = element.split('=')
					self.instanceConfigOp(opType.rstrip(), instance.lstrip())
					self.totalInstance += int(instance.lstrip())


		# Confligre the seed value if one is specified
		# default value is none (so it's non-deterministic)
		if fiParams.has_key(Fields.Seed.value):
			self.faultSeed = np.int32(fiParams[Fields.Seed.value])		
		else:
			self.faultSeed = None  

		# Configure the skip count if one is specified
		# default value is 0
		if fiParams.has_key(Fields.SkipCount.value):
			self.skipCount = np.int32(fiParams[Fields.SkipCount.value])
		else:
			self.skipCount = 0
	# End of constructor

# End of class FIConfig

# These are called from within modifyGraph to read the fault params in a file

def staticFaultParams():
	"Statically hardcoded parameter values for testing"
	params = { }

	# Configure the fault types for Scalars and Tensors	
	params[Fields.ScalarFaultType.value] = FaultTypes.RAND.value		# Scalar Fault type
	params[Fields.TensorFaultType.value] = FaultTypes.RAND.value		# Tensor Fault type

	# List of Operations to fault inject and their probabilities
	params[Fields.Ops.value] = [ "ADD = 0.5", "MUL = 1.0" ]

	# Random seed for the fault injector		
	params[Fields.Seed] = 100000		# Random seed value
	
	# How many operator counts to skip (typically for training)
	params[Fields.SkipCount.value] = 1		# SkipCount value

	# Make sure the parameter dict is returned back to the caller
	return params

def yamlFaultParams(pStream):
	"Read fault params from YAML file"
	# NOTE: We assume pStream is a valid YAML stream
	params = yaml.load(pStream)
	return params

def configFaultParams(paramFile = None):
	"Return the fault params from different files"
	if paramFile == None:
		return staticFaultParams()

	params = {}
	try:
		paramStream = open(paramFile, "r")
	except IOError:
		print "Unable to open file ", paramFile
		return params

	# Check if the file extension is .yaml, and if so parse the Stream 
	# (right now, this is the only supported format, but can be extended)
	if paramFile.endswith(".yaml"):
		params = yamlFaultParams(paramStream)
	else:
		print "Unknown file format: ", paramFile
		
	#print params
	return params
