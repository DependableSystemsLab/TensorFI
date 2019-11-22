# This file is used to store all of the operations that will be tested by regression_operators.py
# By default this should contain all the operations currently supported by TensorFI
# When implementing a new operation in TensorFI, add an entry to the list below for that operation
# Each entry should be a list: [op_name, op_type, input_types]
#   op_name: (String) Corresponding to the operation's entry in the Ops enum class in fiConfig.py (for reference)
#   op_type: (String) The op_type that is passable to the tf.create_op(op_type, inputs) function, i.e., the "type" property of the Operation object (op.type) and should be same as the opTable entry in injectFault.py
#   input_types: (List) The data types the operation supports as inputs, simplified to either "float", "int", "complex", "string", "bool", or "resource" (refer to the tensorflow documentation for each operation)

op_list = [
#["ASSIGN"],
#["IDENTITY"],
["ADD", "Add", ["float", "int", "complex", "string"]],
["SUB", "Sub", ["float", "int", "complex"]]
#["MUL"],
#["SQUARE"],
#["SHAPE"],
#["SIZE"],
#["FILL"],
#["FLOOR-MOD"],
#["RANGE"],
#["RANK"],
#["SUM"],
#["MATMUL"],
#["ARGMAX"],
#["ARGMIN"],
#["EQUAL"],
#["NOT-EQUAL"],
#["LESS-EQUAL"],
#["CAST"],
#["MEAN"],
#["COUNT-NONZERO"],
#["RESHAPE"],
#["CONV2D"],
#["RELU"],
#["MAX-POOL"],
#["SOFT-MAX"],
#["MAXIMUM"],
#["MINIMUM"],
#["EXPAND-DIMS"],
#["SWITCH"],
#["GREATER"],
#["NEGATIVE"],
#["POW"],
#["REALDIV"],
#["ABSOLUTE"],
#["RSQRT"],
#["LOG"],
#["BIASADD"],
#["SIGMOID"],
#["TANH"],
#["PACK"],
#["UNPACK"]
]

