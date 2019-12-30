# TensorFI Test Suite

## Overview

This test suite is comprised of a master script [runAll.py] that calls all of the test scripts included in the suite. First, the setup is tested for correct installations (python version, TensorFI, tensorflow, other python packages). Next, the other test scripts are called to run individual test scripts.

Only stable and functioning test scripts are included in this suite.

## When adding support for a new Operation in TensorFI

After adding TensorFI support for a new Operation, this test suite must be updated to support the added Operation (otherwise the Operation will not be tested). The [operations_runTests.py] file tests the supported Operations to ensure the graphs instrumented by TensorFI result in the same outputs as the uninstrumented versions. 

The inputgen data structure in [operations_inputgen] defines the Operations that will be tested with a mapping to an input generation function. The inputgen functions automatically generate the test input cases for each corresponding Operation. 

When adding support for a new Operation, you must first create a new inputgen function in [operations_inputgen] to define the test cases for that Operation. It is possible to map to an existing inputgen function if the desired input test cases are the same as another existing Operation, otherwise you must create a new inputgen function. 

Please refer to the existing examples for writing the inputgen functions. In general, each input test case is a list of arguments (usually tensors), and all of these input test cases are stored in another list and returned by the function.