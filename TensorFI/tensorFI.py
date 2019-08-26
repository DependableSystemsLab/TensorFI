#!/usr/bin/python

# Library for performing fault injections into TensorFlow graphs (in place)
# Uses the fault injection functions defined in modifyGraph and ficonfig

import tensorflow as tf
import printGraph as pg
import modifyGraph as mg
from injectFault import initFIConfig, initFILog, logRun
from fiConfig import configFaultParams 
import logging
import fiStats
from multiprocessing import Process
from threading import Thread
from math import ceil, floor

# Master fault injector class

class TensorFI:
	"This is the master tensorFI class which has the externally callable functions"
	
	# This is a closure so it can remember the oldRun function and also keep a cache
	def monkeyPatch(self, oldRun, maintainCache = True):
		"Modify the graph and the session to call the corresponding fi nodes upon a run"
		
		# Local state that's specific to the function
		cache = { }	# Previous invocations of the run function
		self.runCount = 0	# Count of how many times run is invoked

		# NOTE: We allow tensorList to be None as it can be loaded from cache
		# It also accepts an additional useCached parameter to load values from cache
		def newRun(tensorList = None, feed_dict=None, useCached = False):
			"Replacement for session's run call - calls fiNodes if needed"
			# NOTE: We should have  called initFIConfig before the run function
			
			if maintainCache:	# If we wish to maintain a cache at runtime
				if useCached:
					# Use a cached value (from the last run)
					tensorList = cache["list"]
					feed_dict = cache["dict"]
				else:
					# Remember this run's values for future FI runs perhaps
					cache["list"] = tensorList	# Keep track of prior invocation list
					cache["dict"] = feed_dict	# Keep track of the prior feed_dict	
		
			# TensorList must not be None at this point as it'd have been loaded from cache
			if tensorList == None: raise ValueException("tensorList cannot be None")

			# If injections are off, don't do any injections
			# So call the old run function which is fast (not instrumented)
 			if self.noInjections:
				logging.info("No injections: Calling oldRun on " + str(tensorList) )
				return oldRun(tensorList, feed_dict)

			# Otherwise, start the fault injector
			self.runCount = self.runCount + 1
			logging.info("Calling newRun " + self.name + " runCount = " + str(self.runCount))
			logRun(self.runCount)	

			# We could have a single tensor, variable or tensorList
			# So check if it is iterable first. If not, handle it separately

			nonIterable = False	# Is the resultList iterable ?
			try:
				# This is the only thing that works across TF versions
				# Other ways of checking if it's an iterator break in some versions
				# 
				for tensor in tensorList: pass
			except Exception:
				# Ok, it's not iterable as it turns out
				# Now append it to a list which will be iterable
				logging.info( str(tensorList) + " is not iterable")
				tensor = tensorList
				tensorList = [ ]
				tensorList.append(tensor)
				nonIterable = True
		
			# Now we have a list of tensors that is also iterable
			# So go over each one and call the fi operation correspoding to it
			results = []
			logging.debug("TensorList = " + str(tensorList))
			for tensor in tensorList:
				# If we have replaced it with a FI function during our traversal
				# logging.debug("Looking up " +  str(tensor.name) + " in fiMap " + fiMap)
				if self.fiMap.has_key(tensor):
					fiTensor = self.fiMap[ tensor ]	 
				else:
					fiTensor = tensor
				logging.debug("Calling oldRun on " +  str(fiTensor.name))
				# logging.debug("feed_dict = " + str(feed_dict))
				
				# if we see an error, catch it and continue with the other tensors
				# FIXME: We should provide some degree of roll-back and retry here
				try:
					# Call the old Run method with the fault injection Tensor 
					res =  oldRun(fiTensor, feed_dict)	
				except Exception as e:
					logging.error("Encountered exception " + str(e))
					logging.error("Unable to execute run on " + str(fiTensor))
					res = None	
					# continue				
				
				# Add the result to the results list
				results.append(res)

			# End for loop
			logging.info("Done with newRun " + self.name)
		
			# If the results list was not iterable, return it directly
			if nonIterable: 
				return results[0]
	
			# Return the results as a TensorList 
			return results	
		# Done with newRun function body

		# Return the newRun function so it can be called in lieu of the old one
		return newRun
		
		# End of monkeyPatch function

	# These are utility functions for debugging

	def printGraph(self):
		"Utility function to print the graph after it is modified"
		print "Graph after instrumentation : "
		pg.printGraph( self.session )

	def printInjectMap(self):
		"Utility function to print the FIMap after modifyGraph is done"
		print "fiMap: "
		for (tensor, fiTensor) in self.fiMap.iteritems():
			print "\t", tensor.name, " : ", fiTensor.name
		print "Done fiMap"

	# These are all functions that can be called externally for interacting with the injector

	def turnOffInjections(self):
		"Turn off fault injections globally"
		logging.info("Turning off injections")
		self.noInjections = True

	def turnOnInjections(self):
		"Turn on fault injections globally"
		logging.info("Turning on injections")
		self.noInjections = False

        # Functions to get and set the logging level - cn be called externally

        def getLogLevel(self):
                "Return the current logging level"
                return logging.getLogger().getEffectiveLevel()

        def setLogLevel(self, level):
                "Set the current log level"
                # Assume that logging.basicConfig has been called already
                logging.getLogger().setLevel(level)

	# This is the externally callable "constructor" function  
	# to instrument a session and monkey patch its run function
	# Optionally takes a fault config file, and DEBUG level as arguments
	# It inserts the FI nodes in the graph and overrides the old run function

	def __init__(self, s,	# This is the session from tensorFlow 
			configFileName = "confFiles/default.yaml",	# Config file for reading fault configuration 
			logDir = "faultLogs/",				# Log directory for the Fault log (Not to be confused with the logging level below)
			logLevel = logging.DEBUG,			# Logging level {DEBUG=10, INFO=20, ERROR=30}
			disableInjections = False,			# Should we disable injections after instrumenting ?
			name = "NoName", 				# The name of the injector, used in statistics and logging
			fiPrefix = "fi_"):				# Prefix to attach to each node inserted for fault injection
		"Initialize the fault injector with the session object"
		 
		self.session = s
		self.name = name

		# Setup the logging level for debug messages
		logging.basicConfig()
                self.setLogLevel(logLevel)
                logging.debug("Done setting logLevel to " + str(self.getLogLevel()) )
	
		# Read the config file parameters from the configuration file
		# If the configFileName is None, it'll use defalt parameters
		logging.info("Initializing the injector")
		fiParams = configFaultParams(configFileName)
	
		# Modify the entire graph to insert the FI nodes - store in fiMap
		logging.info("Modifying graph in session " + s.sess_str)
		graph = s.graph
		self.fiMap = mg.modifyNodes(graph, fiPrefix)
		logging.info("Done modification of graph")
		#self.printInjectMap()	

		# Configure the fault injection parameters for the injection functions	
		# fiConf is a global variable as it needs to be accessible to the FI functions
		logging.info("Initializing the fault injection parameters")
		initFIConfig(fiParams)

		# Initialize the default fault log - this may be overridden by the launch method later
		# This is in case the run method is called directly without going through the launch
		logging.info("Initializing the fault log")
		self.logDir = logDir
		initFILog(self.logDir + self.name)

		# Then, monkey patch the run function of the session 
		# by passing the injection map for it to "remember"
		logging.info("Performing monkey patching")
		self.oldRun = s.run
		s.run = self.monkeyPatch( self.oldRun )

		# Start doing injections unless disableInjections is set to True 	
		self.noInjections = disableInjections
		logging.info("Done with init")
	# Done with init
	
	# This is also an externally callable function to perform fault injections at runtime
	# Parameters: 
	#	numberOfInjections - number of injections to be performed
	#	computeDiff - function to calculate the diff from the correct output of the fault injected one
	#	collectStats(optional) - to collect the statistics (default stats are used otherwise)		
	#	myID(optional)	- to specify the run ID of the fault injection campaign (set to 0 by default)
	#
	def launch(self, numInjections, computeDiff, collectStats = fiStats.defaultStats, myID = 0):
		"Call the run method repeatedly with the last used parameters, and record the results" 	
	
		# NOTE: We need to have called instrumentSession before this call or else it won't work
		# we also assume that at least one run call has been executed AFTER the call to instrumentSession
		# If the second requirement is not met, call run instead (see below) and NOT this function
		# Also , computeDiff should "remember" the correct value and compare with the fault injected one
		
		# This initalizes the statistics to be collected (only if collectStats is not specified)
		collectStats.init()

		# Initialize the fault log, even if it has been initialized earlier in __init__
		# As we may be launching this from multiple processes and we need one log per process
		# logging.info("Initializing the fault injection logs for process " + str(myID))
		# initFILog(self.logDir + self.name, myID)
	
		self.turnOnInjections()	# Enable injections if they've been disabled previously

		logging.info("Starting " + str(numInjections) + " injections: Process " + str(myID))
		# logging.info("computeDiff = " + str(computeDiff))
		# logging.info("collectStats = " + str(collectStats))
		
		for i in range(numInjections):
			logging.info("Experiment : " + str(i) + " Process: " + str(myID))
			
			# Increment the number of injections
			collectStats.update(fiStats.Stats.Injected)

			# Call the original run function with the parameters used earlier in the session
			# Assumption: tensorList and feed_dict parameters have not been modified in between 
			result = self.session.run( useCached = True )
			logging.info("\tResult = " + str(result) )
			
			# Use the provided function to calculate the diff and updateStats based on it
			diff = computeDiff( result )
			
			# Add the difference to the collectStats
			if diff>0:
				collectStats.update(fiStats.Stats.Incorrect)
				collectStats.update(fiStats.Stats.Diff,diff)
				
			# Done with this run
			logging.info("Done experiment " + str(i) + " Process: " + str(myID))
		# End for loop
		
		logging.info("Done running injections for process " + str(myID))
		# logging.info( collectStats.getStats() )

	# Done with launch

	# This is the method to use for FI runs if we haven't called run yet
	#	It also gets the golden output from the correct run
	# 	and then calls fiLaunch with the corresponding function
	#	It takes the following parameters:
	#		numInjections - number of fault injections to be performed
	# 		createDiff - creates a difference function that remembers the correct output
	#				and returns a function to calculate diff with faulty outputs
	#		tensorList - list of tensors on which the original graph should be run
	#		feed_dict - list of dictionary values to feed to the original graph computation
	#
	def run(self, numInjections, createDiff, tensorList, feed_dict, collectStats = fiStats.defaultStats):
		"First call the session's run method, get the correct output, and then do the fault injections"
	
		# Get the Golden output and cache the invocation values
		self.turnOffInjections()
		correct = self.session.run(tensorList, feed_dict)

		# Now perform the fault injections with FILaunch
		self.turnOnInjections()
		self.launch(numInjections, createDiff(correct), collectStats)
	
	# Done with fiRun

	# This is the parallel version of the launch function that launches the fiRuns in parallel 
	# Parameters:
	#	numberOfInjections - number of injections to be performed
	#	numberOfProcesses - number of parallel processes to be launched 
	#				(the injections are divided uniformly among the launched processes)
	#	computeDiff - function to calculate the diff from the correct output of the fault injected one
	#	collectStatsList - to collect the statistics for each process that is launched	
	#			Unlike the sequential launch method, this takes an array of statistics gatherers
	#			one per launched process. The overall statistics needs to be collated later.
	#	parallel (Optional) - Boolean flag to control if parallel launch is done (default = True)
	#	useProcesses (Optional) - Boolean flag to control if processes should be used instead of threads
	#				(default = False). Note this takes effect if and only if parallel is True 
	#	timeout (Optional) - The maximum amount of time to wait for a thread before killing it (default=None)	
	#	
	def pLaunch(self, numberOfInjections, numberOfProcesses, computeDiff, collectStatsList, parallel = True, 
			useProcesses = False, timeout = None):
		"Launches the fault injection runs in parallel using either Threads or Processes"

		# Fist, calculate how many faults to inject in each process
		numInjectionsPerProcess = int( floor( numberOfInjections / numberOfProcesses ) )
		extraInjections = numberOfInjections - numInjectionsPerProcess * numberOfProcesses

		# Now launch the injections in parallel if the parallel flag is set
 
		processes = [ ]		# List of processes or Threads depending on the value of useProcesses		
	
		for i in range(numberOfProcesses):
			
			# Get the stat corresponding to the process (assume that StatsList.length == numProcesses)
			collectStat = collectStatsList[i]
			
			# If it's the last process, only launch as many injections as needed
			if (i == numberOfProcesses - 1): numInjectionsPerProcess += extraInjections
			
			# Now launch the injections and collect the statistics (either sequentially or in parallel)
			if not parallel:
				self.launch( numInjectionsPerProcess, computeDiff, collectStat )
			else:
				# If we enable parallel launch, we need to pack the arguments into an array
				argArray = []
				argArray.append( numInjectionsPerProcess )
				argArray.append( computeDiff )
				argArray.append( collectStat )
				argArray.append( i )
				logging.info("Launching process " + str(i) + " with arguments " + str(argArray))
				if useProcesses:
					# FIXME: TensorFlow hangs when we use processes, so beware !
					p = Process( target = self.launch, args = argArray )
				else:
					p = Thread( target = self.launch, args = argArray )
				p.start()
				processes.append(p)
		# End for

		# Wait for all the launched processes to finish
		if parallel:	
			for i in range(numberOfProcesses):
				logging.info("Waiting for process " + str(i))
				# This can terminate either because the thread terminated or timeout
				processes[i].join(timeout)
				# Check if there was a timeout
				if processes[i].isAlive():
					logging.info("Process " + str(i) + " timed out")
					# FIXME: We should wait for the process to terminate naturally
					# as it can leave the timedout process in an unstable state now
				else:
					logging.info("Process " + str(i) + " terminated")
			# End for
		
	# End of pLaunch	

	# This method removes the fault injection capability
	# NOTE: It is not possible to inject faults after this is called
	def doneInjections(self):
		"Unlinking the current session and replaces the patch"
		logging.debug("Unlinking session " + self.name)
		self.session.run = self.oldRun
		self.session = None
	
	# Destructor function
	# NOTE: We don't call doneInjections as it may have been called already
	def __del__(self):
		"Cleaup all the file handles" 
		logging.debug("Destroying injector " + self.name)
		logging.shutdown()
	# Done with the destructor

# Done with the FI class
