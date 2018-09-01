# Simple statistics gathering for fault injection experiments

from enum import Enum
import sys

class Stats(Enum):
	Injected = "Injected"
	Incorrect = "Incorrect"
	Diff = "Diff" 
# End of Stats Enum 

# Initial values of the statistics gathered
initVal = { Stats.Injected: 0,
	    Stats.Incorrect: 0,
	    Stats.Diff: 0.0
}

# NOTE: This is the base class that can be overridden for more elaborate functionality
class FIStat:
	"Base class for statistics gathering"

	def __init__(self, name = "", fileName=None):
		"Setup statistics collection with a default file name"
		self.stats = { }
		self.name = name
		self.outFile = sys.stdout	
		if fileName:
			# FIXME: Open a file and dump stats to it later
			try:
				self.outFile = open(fileName,"w")
			except IOError:
				print "Error opening statistics file", fileName

	def init(self):
		"Initialize the statistics"
		for stat in Stats:
			statName = stat.value
			self.stats[ statName ] = initVal[ stat ]

	def update(self, stat, value = 1):
		"Increment the statistic by the value"
		self.stats[ stat.value ] += value

	def getStats(self):
		"Return the stats dictionary as a string"	
		resStr =  self.name + " { "
		for (key, value) in self.stats.items():
			resStr += "\n\t" + str(key) + " : " + str(value)
		resStr += "\n}"
		return resStr

	def writeToFile(self):
		"Write the statistics to a file"
		self.outFile.write( "-------------------\n")
		self.outFile.write( self.getStats() + "\n" )
		self.outFile.write( "-------------------\n")


	def __del__(self):
		"Destructor: make sure the output file is closed"
		if self.outFile!=sys.stdout:
			self.writeToFile()
			self.outFile.close()
	
# Done with FIStat

defaultStats = FIStat("Default")

def getDefaultStats():
	"Return the default stats as a string"
	return defaultStats.getStats()

def collateStats(StatList, name = "Overall", fileName = None):
	"Takes a bunch of different Statistics and collates them together"
	resultStat = FIStat(name, fileName)
	resultStat.init()
	# We asssume the StatList has only FIStat objects here
	for stat in StatList:
		for statField in Stats:
			statName = statField.value
			value = stat.stats[ statName ]
			resultStat.update( statField, value ) 
		# End inner for
	# End outer for
	return resultStat
