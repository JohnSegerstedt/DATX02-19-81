import os
import pymongo
import os.path
import sys
from datetime import datetime
import subprocess
import multiprocessing
from multiprocessing import Manager, Value, Lock
from itertools import repeat

# Requests sc2reaper to parse all files in <folderToParse>.

# Steps per frame is managed in sc2reaper/sc2reaper/sweeper.py
# Matchup specification (PvP or other), database name and port is managed in sc2reaper/sc2reaper/sc2reaper.py


# ----- IMPORTANT ------------
# In case an early abort was made, reapHelper will conncect to the database and erase any broken files.
databaseName = "vision_testing"

# ----- INTERESTING VARS ------
# The directory containing the replay files, change if needed.
folderToParse = "tmp3"
# Name of directory where rep files end up after parsing, no matter if parsing was successful nor not.
subdir = "parsed/"
processes=2


# ----- SETUP ------
dirToParse = os.path.abspath(folderToParse)
dirInProgress = os.path.join(dirToParse, "inProgress/")
dirParsed = os.path.join(dirToParse, subdir)
fileCounter = 0
totalNrFiles = len(os.listdir(folderToParse))
gamesParsed = []

# Requests sc2reaper to parse the file (via os)


def parse(replayFile):
    global fileCounter
    global totalNrFiles
    global startTime
    fileCounter += 1
    print("requesting parse of file number: ",
          fileCounter, " out of ", totalNrFiles, " ...")
    print("total running time: ", (datetime.now() - startTime))
    print(fileCounter)
    fullPath = os.path.join(dirInProgress, replayFile)
    try:
        print("trying")
        os.system("sc2reaper ingest " + fullPath)
    except:
        print("Something went wrong with " + replayFile)


def moveToParsed(replayFile):
    origin = os.path.join(dirInProgress, replayFile)
    destination = os.path.join(dirParsed, replayFile)
    os.rename(origin, destination)


def moveToInProgress(replayFile):
    origin = os.path.join(dirToParse, replayFile)
    destination = os.path.join(dirInProgress, replayFile)
    os.rename(origin, destination)


def moveTodirToParse(replayFile):
    origin = os.path.join(dirInProgress, replayFile)
    destination = os.path.join(dirToParse, replayFile)
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
    # allCollections = ["actions", "players", "replays", "scores", "states"]

    for collection in db.list_collection_names():
        # delete in collection:
        result = db[collection].delete_many(query)
        print(result.deleted_count, " documents deleted in ", collection)


def parseWithMove(replayFile, startTime, counter, totalReplays):
    print("running", replayFile)
    if (replayFile.endswith('.SC2Replay')):
        moveToInProgress(replayFile)
        parse(replayFile)
        moveToParsed(replayFile)
        counter.value += 1
        print("SUCCESSFULLY PARSED", replayFile, "::", str(counter.value), "/", str(totalReplays), " = ", str(counter.value/totalReplays*100), "% TIME SO FAR:", str(datetime.now() - startTime))


# ----- "MAIN" -----
startTime = datetime.now()
print("running main")
if not os.path.isdir(dirParsed):
    os.makedirs(dirParsed)
if not os.path.isdir(dirInProgress):
    os.makedirs(dirInProgress)
else:
    # If .../inProgress/ contains files, an early abort is assumed. Parsing any files in this folder from scratch.
    for replayFile in os.listdir(dirInProgress):
        if replayFile.endswith('.SC2Replay'):
            print("replay(s) in progress found, removing broken data caused by ",
                  replayFile, "in mongodatabase")
            removeBrokenFileData(replayFile, databaseName)
            print("replay(s) in progress found, moving back to unparsed folder:", replayFile)
            moveTodirToParse(replayFile)

replays = [f for f in os.listdir(folderToParse) if os.path.isfile(os.path.join(folderToParse, f)) and f.lower().endswith(".sc2replay")]
startTime = datetime.now()
manager = multiprocessing.Manager()
counter = manager.Value("i", 0)
nReplays = len(replays)
pool = multiprocessing.Pool(processes)
pool.starmap(parseWithMove, zip(replays, repeat(startTime), repeat(counter), repeat(nReplays)))

print("script finished")
print("total files: ", fileCounter)
print("total running time: ", (datetime.now() - startTime))
