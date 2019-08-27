#!/bin/bash

PYTHON="python"	# Name of the default $PYTHON interpreter
		# On some systems, it's python2, so change it


# NOTE: This must be run from the TensorFI-home directory
# 	and NOT directly from the Tests directory or else
#	it will fail. All paths are relative to TensorFI-home

# Check if Python interpreter is installed
if [[ $($PYTHON -c "import sys" | wc -c) -ne 0 ]]; then
    echo "Cannot find $PYTHON. Aborting ....";
    exit 0
fi

# Check if default $PYTHON version is 3, and if so abort
if  [[ $($PYTHON -c "import sys; print\"\"" | wc -c) -ne 1 ]]; then
  echo "$PYTHON is a Python 3 interpreter - not supported. Aborting...";
  exit 1
fi

# Check if Tensorflow is installed on Python
if [[ $($PYTHON -c "import tensorflow" | wc -c) -ne 0 ]]; then
  echo "Cannot find tensorflow installed for $PYTHON. Aborting..."
  exit 2
fi

# Check if PyYaml is installed on Python
if [[ $($PYTHON -c "import yaml" | wc -c) -ne 0 ]]; then
  echo "Cannot find pyyaml installed for $PYTHON. Aborting..."
  exit 3
fi

# Finally, check if TensorFI is accessible from within Python
if [[ $($PYTHON -c "import TensorFI" | wc -c) -ne 0 ]]; then
  echo "Unable to import TensorFI. Check your PYTHONPATH. Aborting..."
  exit 3
fi

# Now, create the faultLogs and fiStats diretories if they don't already exist
LOGDIR="faultLogs"
if [ ! -d "$LOGDIR" ]; then
	echo "Creating $LOGDIR directory..."
	mkdir "$LOGDIR"
fi

STATDIR="stats"
if [ ! -d "$STATDIR" ]; then
	echo "Creating $STATDIR directory..."
	mkdir "$STATDIR"
fi

# These are the lists of Tests that currently run (with TensorFI)
# FIXME: Automatically generate config files for each Test
# and check their outputs against the predefined config files
 
echo "Starting Tests......."

OUTPUTDIR="/dev/null"	# Directory to which test output is redirected

TIMEOUT_VAL="100s"	# Maximum number of seconds to wait for each test

# Check if timeout command exists, and if not, set TIMEOUT to ""
# NOTE: Timeout doesn't exist on MacOSX unfortunately !
if [[ $(timeout --version | wc -c) -eq 0 ]]; then
	echo "Unable to find Timeout command. Setting it to empty"
	TIMEOUT=""
else
	TIMEOUT="timeout $TIMEOUT_VAL"
fi 

# Suppress TensorFlow warning messages
export TF_CPP_MIN_LOG_LEVEL=1

# Add any tests that should be executed here
Tests=( "noop"
        "constant" 
	"variables" 
	"placeholder" 
	"loss" 
	"gradient" 
	"linear_regression" 
	"logistic_regression" 
	"nearest_neighbor" 
	"mnist_softmax" 
	"keras-mnist"
	"neural_network_raw" 
)

# Run each test and redirect its output to an output dir
# Check if the test returned an exit code, if not, it passed
for test in ${Tests[@]}
do
	echo "Running $test.py..." 
	$TIMEOUT $PYTHON Tests/$test.py 1>"$OUTPUTDIR"
	exitcode=$?
	if [ $exitcode -ne 0 ]; then
		echo "	Test $test.py failed with exit code $exitcode"
	else
		echo "	Test $test.py passed" 
	fi
done

echo "Done with Tests......."
