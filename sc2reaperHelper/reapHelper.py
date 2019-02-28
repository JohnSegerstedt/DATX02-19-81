import os
import os.path
import sys
from datetime import datetime

# Requests sc2reaper to parse all files in <folder>. 

# Steps per frame is managed in sc2reaper/sc2reaper/sweeper.py
# Matchup specification (PvP or other), database name and port is managed in sc2reaper/sc2reaper/sc2reaper.py

# To call this script:
# $ python reapHelper.py <directory>

# Input
folder = str(sys.argv[1]) 

# Vars
dirPath = os.path.abspath(folder)
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
	fullPath = os.path.join(dirPath, replayFile)
	try:
		#print("debug", fullPath)
		os.system("sc2reaper ingest " + fullPath)
		#sc2reaper.ingest(os.path.join(fullPath))
	except:
		print("Something went wrong with " + replayFile)

# --- "MAIN" ---
startTime = datetime.now()
try:
	for replayFile in filesToBeProcessed:
		parse(replayFile)
except:
	print("Couldn't read directory")
print("script finished")
print("total files: ", fileCounter)
print("total running time: ", (datetime.now() - startTime))
