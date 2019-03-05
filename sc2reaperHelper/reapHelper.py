import os
import os.path
import sys
from sc2reaper import sc2reaper
from datetime import datetime

# Requests sc2reaper to parse all files in <folder>. 

# Steps per frame is managed in sc2reaper/sc2reaper/sweeper.py
# Matchup specification (PvP or other), database name and port is managed in sc2reaper/sc2reaper/sc2reaper.py

# ----- INTERESTING VARS ------
folder = "tmp10/" 	# The directory containing the replay files, change if needed.
subdir = "parsed/" 	# Name of directory where rep files end up after parsing, no matter if parsing was successful nor not.

# ----- SETUP ------
dirToParse = os.path.abspath(folder)
dirParsed = os.path.join(dirToParse, subdir)
fileCounter = 0
filesToBeProcessed = os.listdir(folder)
totalNrFiles = len(filesToBeProcessed)

# Requests sc2reaper to parse the file (via os)
def parse(replayFile):
	global fileCounter
	global totalNrFiles
	global startTime
	fileCounter += 1
	print("requesting parse of file number: ", fileCounter, " out of ", totalNrFiles, " ...")
	print("total running time: ", (datetime.now() - startTime))
	print(fileCounter)
	fullPath = os.path.join(dirToParse, replayFile)
	try:
		#print("debug", fullPath)
		os.system("sc2reaper ingest " + fullPath)
		#sc2reaper.ingest(os.path.join(fullPath))
	except:
		print("Something went wrong with " + replayFile)

def moveFile(file):
	origin = os.path.join(dirToParse, replayFile)
	destination = os.path.join(dirParsed, replayFile)
	os.rename(origin, destination)


# ----- "MAIN" -----
startTime = datetime.now()

if not os.path.isdir(dirParsed):
        os.makedirs(dirParsed)
try:
	for replayFile in filesToBeProcessed:
		if replayFile.endswith('.SC2Replay'):
			parse(replayFile)
			moveFile(replayFile)
except:
	print("Couldn't read directory")
print("script finished")
print("total files: ", fileCounter)
print("total running time: ", (datetime.now() - startTime))
