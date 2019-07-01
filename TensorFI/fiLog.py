# Keeps track of the global fault log for posteriori analysis and debugging

from enum import Enum
import datetime
import sys


# Enum for the fields to write in the log file
# If you want to add a new field, add it here first
# Likewise, if you want to change the names of the fields, do it here
class Log(Enum):
	RunCount = "RunCount"
	OpName = "OpName"
	Count = "Count"
	Original = "Original"
	Injected = "Injected"
	TimeStamp = "Time"
	Instance = "Instance"
	TotalInstance = "TotalInstance"
# End of LogEntries Enum

class FILog:
	"Keep track of the fault injection log and writes to it"

	def __init__(self, name = "default"):
		"Open the file for recording fault injection logs"
		fiName = name + "-log"
		try:
			self.logFile = open(fiName, "w")
		except IOError:
			print "Unable to open log file", fiName
			self.logFile = sys.stdout
		self.logEntry = {}
		self.startTime = self.getTimeStamp()
		self.logFile.write("Starting log at " + str(self.startTime) + "\n")
		self.blankLine()

	def dashedLine(self):
		"Write a line of dashes to the log file"
		self.logFile.write("\n---------------------------------------\n")

	def blankLine(self):
		self.logFile.write("\n")

	def updateOp(self, op):
		"Update the operation being injected"
		self.logEntry[Log.OpName] = op 

	def updateCount(self, count):
		"Update the operation count being injected"
		self.logEntry[Log.Count] = count

	def updateInjected(self, injectedVal):
		"Update the injected value"
		self.logEntry[Log.Injected] = injectedVal

	def updateOriginal(self, originalVal):
		"Update the original value"
		self.logEntry[Log.Original] = originalVal

	def updateTimeStamp(self, timeVal):
		"Update the time stamp based on difference from start time"
		diffTime = (timeVal - self.startTime)
		# FIXME: This won't work if the injection spans multiple days
		self.logEntry[Log.TimeStamp] = str(diffTime)

	def updateRunCount(self, runCount):
		"Update the overall runCount"
		self.logEntry[Log.RunCount] = runCount

	def updateInjectedInstance(self, instance, totalInstance):
		"Update the instance to be injected"
		"Note that the instance may vary based on the injection mode chosen"
		"In the 2nd injection mode: the instance is based on the overall instance of current op"
		"In the 3rd injection mode: in the isntance is based on the ALL the op in the algorithm"
		self.logEntry[Log.Instance] = instance
		self.logEntry[Log.TotalInstance] = totalInstance

	def getLogEntry(self):
		"Get a string representation of the log entry"
		res = [ "{" ]
		for (key, value) in self.logEntry.items():
			res.append( "\t" + str(key.value) + " : " + str(value) )
		res.append(" }")
		return "\n".join(res)

	def getTimeStamp(self):
		"Return the current time in seconds"
		return datetime.datetime.now()

	def commit(self):
		"Write the log entry to the log file"
		self.updateTimeStamp( self.getTimeStamp() )
		self.logFile.write( self.getLogEntry() )
		self.blankLine()
		self.logFile.flush()

	def __del__(self):
		"Close the logFile"
		self.dashedLine()
		self.logFile.write("Done injections\n")
		self.logFile.close()

# End of class FILog
