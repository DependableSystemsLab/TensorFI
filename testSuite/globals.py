
# Set your directory to redirect test outputs and logging
# Default is "/dev/null" (suppress output and logs)
OUTPUTDIR = "/dev/null"
LOGDIR = "/dev/null"

# Set global variable for access to the script paths and folders
import os
TESTSUITE_DIR   = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))
TENSORFI_ROOT   = os.path.abspath(os.path.join(TESTSUITE_DIR, os.pardir))
TENSORFI_SOURCE = os.path.abspath(os.path.join(TENSORFI_ROOT, "TensorFI"))
