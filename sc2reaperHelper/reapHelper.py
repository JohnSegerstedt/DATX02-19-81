import os
import pymongo
import os.path
import sys
from datetime import datetime

# Requests sc2reaper to parse all files in <folderToParse>. 

# Steps per frame is managed in sc2reaper/sc2reaper/sweeper.py
# Matchup specification (PvP or other), database name and port is managed in sc2reaper/sc2reaper/sc2reaper.py


# ----- IMPORTANT ------------
#In case an early abort was made, reapHelper will conncect to the database and erase any broken files.
databaseName = "vision_testing"

# ----- INTERESTING VARS ------
folderToParse = "tmp3" 	# The directory containing the replay files, change if needed.
subdir = "parsed/" 	# Name of directory where rep files end up after parsing, no matter if parsing was successful nor not.


# ----- SETUP ------
dirToParse = os.path.abspath(folderToParse)
dirInProgress = os.path.join(dirToParse, "inProgress/")
dirParsed = os.path.join(dirToParse, subdir)
fileCounter = 0
totalNrFiles = len(os.listdir(folderToParse))

# Requests sc2reaper to parse the file (via os)
def parse(replayFile):
	global fileCounter
	global totalNrFiles
	global startTime
	fileCounter += 1
	print("requesting parse of file number: ", fileCounter, " out of ", totalNrFiles, " ...")
	print("total running time: ", (datetime.now() - startTime))
	print(fileCounter)
	fullPath = os.path.join(dirInProgress, replayFile)
	try:
		#print("debug", fullPath)
		os.system("sc2reaper ingest " + fullPath)
		#sc2reaper.ingest(os.path.join(fullPath))
	except:
		print("Something went wrong with " + replayFile)

def moveToParsed(file):
	origin = os.path.join(dirInProgress, replayFile)
	destination = os.path.join(dirParsed, replayFile)
	os.rename(origin, destination)

def moveToInProgress(file):
	origin = os.path.join(dirToParse, replayFile)
	destination = os.path.join(dirInProgress, replayFile)
	os.rename(origin, destination)

def moveTodirToParse(file):
	origin = os.path.join(dirInProgress, replayFile)
	destination = os.path.join(dirInProgress, replayFile)
	os.rename(origin, destination)

def connectToDatabase(db):
  # Create mongodb client
  client = pymongo.MongoClient()
  # Connect to db using client
  db = client[db]
  return db

def removeBrokenFileData(file, databaseName):
	db = connectToDatabase(databaseName)
	query = {"replay_name": os.path.join(dirInProgress, replayFile)}
	#allCollections = ["actions", "players", "replays", "scores", "states"] 
	
	for collection in db.list_collection_names():
		#delete in collection:
		result = db[collection].delete_many(query)
		print(result.deleted_count, " documents deleted in ", collection)

# ----- "MAIN" -----
startTime = datetime.now()

if not os.path.isdir(dirParsed):
        os.makedirs(dirParsed)
if not os.path.isdir(dirInProgress):
        os.makedirs(dirInProgress)
else:	
	# If .../inProgress/ contains files, an early abort is assumed. Parsing any files in this folder from scratch. 
	for replayFile in os.listdir(dirInProgress):
		if replayFile.endswith('.SC2Replay'):
			print("replay(s) in progress found, removing broken data caused by ", replayFile, "in mongodatabase")
			removeBrokenFileData(replayFile, databaseName)
			print("replay(s) in progress found, parsing " , replayFile)
			#parse(replayFile)
			moveToParsed(replayFile)

for replayFile in os.listdir(folderToParse):
	if replayFile.endswith('.SC2Replay'):
		moveToInProgress(replayFile)
		parse(replayFile)
		moveToParsed(replayFile)

print("script finished")
print("total files: ", fileCounter)
print("total running time: ", (datetime.now() - startTime))
