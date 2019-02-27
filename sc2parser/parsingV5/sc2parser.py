import sys
import replay_parser
from pathlib import Path
import os
import datetime

# To call this script use these kinds of formats:
# $ python "C:\Program Files\Git\src\sc2reader_scripts\sc2parser.py" replaysTest 0 Replays3

# --- INPUT VARIABLES ---     
source = str(sys.argv[1])       # either folder or .txt-file
time = float(sys.argv[2])       # when to parse in real seconds, <1.0 --> SUPER PARSE
destination = str(sys.argv[3])  # the name of .csv-file to put data

# --- TOUCHABLE VARIABLES ---
currentTime = time
startTime = 60
stepSize = 30
maxTime = 240
stepsNeeded = (maxTime-startTime) / stepSize
path = "replays\\"


# --- UNTOUCHABLE VARIABLES --- 
counter = 0
maxAmount = 0
fileFlag = -1            # 0=.txt, 1=folder

# --- FUNCTIONS ---
def printStuff(replayIndex, replayTitle):
        currentDT = datetime.datetime.now()
        endDateTime = datetime.datetime.now()
        timeDifference = endDateTime - startDateTime
        print("Replay:"+str(replayIndex)+"/"+str(int(maxAmount))+" - ReplayName:"+replayTitle+" - Progress:["+str(int(100*counter/maxAmount))+"%] - Time:"+currentDT.strftime("%Y-%m-%d %H:%M")+" - TIME ELAPSED:"+str(timeDifference).split('.', 2)[0])
def parseTxtFile(start, interval, end):
        with open(source) as f:
                for line in f:
                        global counter
                        counter += 1
                        replayName = line[:-1]
                        replayName = path + replayName
                        printStuff(counter, replayName)
                        replay_parser.mainFunction(replayName, start, interval, end, destination)
def parseFolder(start, interval, end):
        for filename in os.listdir(source):
                global counter
                counter += 1
                replayName = source+"/"+filename
                printStuff(counter, replayName)
                replay_parser.mainFunction(replayName, start, interval, end, destination)
# --- "MAIN" ---
startDateTime = datetime.datetime.now()
print ("----------INITIALING PARSING AT:"+startDateTime.strftime("%Y-%m-%d %H:%M")+"----------")
if(source.lower().endswith('.txt')):
        fileFlag = 0
else:
        fileFlag = 1
if(time < 1.0):
        if(fileFlag == 0):
                with open(source) as f:
                        for line in f:
                                maxAmount += 1
                parseTxtFile(startTime, stepSize, maxTime)
        if(fileFlag == 1):
                for filename in os.listdir(source):
                        maxAmount += 1
                parseFolder(startTime, stepSize, maxTime)
else:
        stepsNeeded = 1
        if(fileFlag == 0):
                with open(source) as f:
                        for line in f:
                                maxAmount += 1
                parseTxtFile(currentTime, 0, currentTime)
        if(fileFlag == 1):
                for filename in os.listdir(source):
                        maxAmount += 1
                parseFolder(currentTime, 0, currentTime)
endDateTime = datetime.datetime.now()
timeDifference = endDateTime - startDateTime
print ("----------INITIALIZED AT:"+startDateTime.strftime("%Y-%m-%d %H:%M")+" - FINISHED AT:"+endDateTime.strftime("%Y-%m-%d %H:%M")+" - TIME ELAPSED:"+str(timeDifference).split('.', 2)[0]+"----------")
