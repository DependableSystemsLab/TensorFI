# ModifyGraph - has the functions to instrument and modify the TensorFlow graph

import tensorflow as tf
import printGraph as pg
import injectFault
import numpy as np

def createFIFunc(opType, inputs, outputTypes, name):
	"Create a tensorflow operation representing a fault injection node"
	# print "\nCreating FIfunc with ", opType, inputs, outputTypes, name

	fiFunc = None

	# Check the opType and return the corresponding function as Fifunc
	if opType=="Cast":
		# We have to special case Cast as it's expected to "remember" its type
		# This could be due to a bug in TensorFlow (at least it's not documented)	
		fiFunc = injectFault.createInjectFaultCast( outputTypes[0] )

	elif injectFault.opTable.has_key( opType ):
		# Lookup the opTable and return the corresponding function (injectFault...)
		# This is the default case if there's an injectFault for the function 
		fiFunc = injectFault.opTable[ opType ]
	else: 
		# It's not a known operation, so use the generic injection function
		fiFunc = injectFault.opTable[ "Unknown"]
		#pass
	
	# fiFunc should have been initialized (fiFunc != None)
	if fiFunc == None: 
		raise ValueError("Unknown operation : " + str(opType))	

	# Create a new TensorFlow operator with the corresponding fault injection function
	res = tf.py_func(fiFunc, inputs, outputTypes, name = name) 

	#print "NewOp = ", res

	return res
# Done with createFIFunc

# Create fault injection equivalents for everything except {Placeholder, Variable, Constant, NoOp}
def excludeOps(op):
        "Which operations to exclude from the instrumentation"
        result = False
        result = result or op.type=="Placeholder"
        result = result or op.type.startswith("Variable") 
        result = result or op.type=="Const"
        result = result or op.type=="NoOp"
        return result
# Done with excludeOps

def modifyNodes(g, prefix):
	"Insert nodes in the graph for fault injection corresponding to the original nodes"
	ops = g.get_operations()

	fiMap = {} # Keeps track of the mapping between the FI node inserted and the original ones


	# Iterate over all the nodes in the TensorFlow graph
	for op in ops:
		# print("Modifying: " + pg.getOperation(op) )
		# Gather all the inputs in a list, replacing them with those from fiMap
		inputs = []
		for input in  op.inputs:
			if fiMap.has_key(input): 
				input = fiMap[input]
			inputs.append(input)
		 
		# important attributes (e.g., strides) will be used as as the input of the tensor as well
		# Please check if the Op you want to inject has these attributes, if any, you should provide these to the customized Op for injection as well
		if(op.type=="Conv2D"):  
			inputs.append( (op.node_def.attr['strides'].list.i[:])  )
			inputs.append( str(op.node_def.attr['padding'].s)  )  
		elif(  op.type=="LRN" ):
			inputs.append( float(op.node_def.attr['bias'].f) )
			inputs.append( float(op.node_def.attr['alpha'].f) )
			inputs.append( float(op.node_def.attr['beta'].f) ) 
		elif(  op.type == "MaxPool"): 
			inputs.append( np.asarray(op.node_def.attr['ksize'].list.i[:]) )
			inputs.append( np.asarray(op.node_def.attr['strides'].list.i[:]) )
			inputs.append( str(op.node_def.attr['padding'].s) ) 
				

		# Create a new operation by wrapping debugPrint function and setting its inputs to op.inputs
		# Do this while preserving the control dependendencies of the original
		with g.control_dependencies(op.control_inputs):
			name = prefix + op.name

		        # If operation is not one of the excluded operations for fault injections
                        if not excludeOps(op):	

				# Find the output types of all the outputs of op	 		
				outputTypeList = []
				for output in op.outputs:
					outputTypeList.append(output.dtype)

				# Create a new fault injection operation with the same inputs and outputs
				newOp = createFIFunc(op.type, inputs, outputTypeList, name)

				# Add newOp's output to the fiMap hashtable for each output of the current node
				# This will be used later to replace downstream operations that depend on it
				for i in range(0, len(op.outputs)):
					output = op.outputs[i]
					fiMap[output] = newOp[i]
				# Done with inner loop
			# Done if
		# Done with 	
	# Done with the outerloop loop 
	return fiMap
# Done with modifyNodes

