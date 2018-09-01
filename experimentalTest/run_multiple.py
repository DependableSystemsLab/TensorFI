import sys, os

# This is the test for running a specific test for multiple times
# Specify the following args: 1) number of runs; 2) path for the tests (i.e., path for the .py file); 
# 3) path for the file to log the accuracy results.

runNum = int(sys.argv[1])
runPath = sys.argv[2]
logPath = sys.argv[3]
###

os.system("rm " + logPath)

for x in range(0, runNum):
	os.system("python " + runPath + " " + logPath)
