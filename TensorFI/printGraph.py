#!/usr/bin/python
# PrintGraph library for iterating over and pretty-printing a TensorFlow graph

def getTensor(tensor):
	"Returns a string representation of a tensor"
	result = ["{"]
	result.append( tensor.name )
	result.append( tensor.dtype.name )
	if tensor.shape:
		shapeList  = tensor.shape.as_list()
		shapeStr = ""
		for shape in shapeList:
			shapeStr = shapeStr + str(shape)
 		result.append(shapeStr)
	result.append("}")
	return " ".join(result)


def getOperation(op):
	"Returns a specific operation as a string"
	opAttr = ["{"]
	opAttr.append("type: " + op.type)
	opAttr.append("name: " + op.name) 
	opAttr.append("inputs { ")
	for input in op.inputs:
		tensorStr = getTensor(input)
		opAttr.append( tensorStr )
	opAttr.append("}")
	opAttr.append("control_inputs { ")
	for control_input in op.control_inputs:
		tensorStr = getOperation(control_input)
		opAttr.append( tensorStr )
	opAttr.append("}")
	opAttr.append("outputs: { ")
	for output in op.outputs:
		tensorStr = getTensor(output) 
		opAttr.append( tensorStr )
	# opAttr.append( str(op.run) )
	opAttr.append("}")
	sep = "\t"
	return sep.join(opAttr)



def getOperations(g):
	"Return the operations in a graph as a string"
	ops = g.get_operations()
	if len(ops)==0:
		return "{ }\n"
	str = "{\n"
	for op in ops: 
		opStr = getOperation(op)
		str = str + "\t" + opStr + "\n"
	str = str + "}\n";
	return str

def getGraph(s):
	"Returns the operations of the graph in the session"
	g = s.graph
	result = [ ]
	if g is not None:
		result.append( "Version : " + str(g.version) )
		result.append( "Name scope : " + g.get_name_scope() )
		result.append( "Graph: " + getOperations(g) )
	return "\n".join(result)

def printGraph(s):
	"Print the graph corresponding to session s"
	print( getGraph(s) )	
