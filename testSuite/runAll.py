#!/usr/bin/python

# Script to run the test suite
# Please include only stable and working tests in this file

# Check correct python version
import sys
if sys.version_info[0] == 2 and sys.version_info[1] >= 7:
    # version is 2.7+ but not 3
    pass
else:
    raise Exception("Please run with python 2.7+ (Python 3 is not supported)")

# Begin checking python package installations
sys.stdout.write("Checking for correct python packages...")
sys.stdout.flush()

# Check that TensorFlow is installed
try:
    import tensorflow
except ImportError as e:
    print "\n\nTensorFlow is not installed properly, unable to import: " + str(e)
    exit()

# Check that PyYaml is installed
try:
    import yaml
except ImportError as e:
    print "\n\nPyYaml is not installed properly, unable to import: " + str(e)
    exit()

# Check that TensorFI is installed
try:
    import TensorFI
except ImportError as e:
    print "\n\nTensorFI is not installed properly, unable to import: " + str(e)
    exit()

# Check python package versions
if tuple(map(int, (tensorflow.__version__.split("."))))[0] < 1:
    print "\n\nTensorFlow version must be 1.0 or newer"
    exit()
if tuple(map(int, (yaml.__version__.split("."))))[0] < 3:
    print "\n\nPyYaml version must be 3.0 or newer"
    exit()

sys.stdout.write("\rChecking for correct python packages... Passed\n")
sys.stdout.flush()

print "\nBeginning test cases...\n"
passed_tests = []
failed_tests = []

# regression operators test
import regression_operators
sys.stdout.write("Running regression_operators test...")
sys.stdout.flush()
if regression_operators.run_test(suppress_out=True)[0]:
    passed_tests.append("regression_operators.py")
    sys.stdout.write("\rRunning regression_operators test... Passed\n")
    sys.stdout.flush()
else:
    passed_tests.append("regression_operators.py")
    sys.stdout.write("\rRunning regression_operators test... Failed\n")
    sys.stdout.flush()
