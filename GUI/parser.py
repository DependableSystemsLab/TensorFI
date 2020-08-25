import sys
 
print("Output from Python") 
print("Config File: " + sys.argv[1])
print("Log File: " + sys.argv[2])
print("TensorFlow Path: " + sys.argv[3])

session = None
runLine = None
multiLine = ""
indentCount = 0
count = 0

def endLine(inputLine):
	global count
	ret = ""

	for char in inputLine:
		ret = ret + char
		if char == ')' and count == 1:
			count = 0
			return ret
		elif char == '(':
			count = count + 1
		elif char == ')':
			count = count - 1
	return None

try:
	f = open(sys.argv[3], "r")
	writeFile = open("runFile.py", "w")

	for line in f:
		if "Session()" in line:
			if "as" in line:
				session = (line.split(" as ")[1]).split(":")[0]
			if "=" in line:
				session = line.split("=")[0]
				session = session.strip()

		if not session is None:
			if (session + ".run(") in line or not multiLine == "":
				inside = None

				if multiLine == "":
					if not endLine(line.split(".run")[1]) is None:
						inside = endLine(line.split(".run")[1])
					else:
						multiLine = multiLine + line.split(".run")[1]
				else:
					saved = endLine(line)
					if saved is None:
						multiLine = multiLine + line
					else:
						multiLine = multiLine + saved
						inside = multiLine
						inside = inside.replace("\n", "")
						inside = inside.replace("  ", "")
						multiLine = ""

				if not inside is None and (len(inside.split(",")) > 1 or runLine is None):
					runLine = session + ".run" + inside

			if ".eval(" in line:
				runLine = line.split('=')[1]

		indentCount = len(line) - len(line.lstrip(' '))
		writeFile.write(line)

	if runLine is None:
		raise Exception("File format not recognized")

	tab = indentCount * " "
	writeFile.write(tab + "import TensorFI as ti\n")
	writeFile.write(tab + "fi = ti.TensorFI(" + session + ", configFileName = \"" + sys.argv[1] + "\", name = \"test\", logLevel = 0, disableInjections = True, logDir = \"" + sys.argv[2] + "\")\n")

	writeFile.write(tab + "fi.turnOnInjections()\n")
	writeFile.write(tab + "fi.turnOnConfig()\n")
        writeFile.write(tab + "acc_no = numpy.around(" + runLine + "[0], decimals=7)\n")
        writeFile.write(tab + "print(\"Accuracy (with config): \" + str(acc_no))\n")
	writeFile.write(tab + "fi.turnOffConfig()\n")

        writeFile.write(tab + "acc_fi = numpy.around(" + runLine + "[0], decimals=7)\n")
        writeFile.write(tab + "print(\"Accuracy (with injections): \" + str(acc_fi))\n")

finally:
	f.close()
	writeFile.close()

