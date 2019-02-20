import sys
import replay_parser
from pathlib import Path

sourceFile = str(sys.argv[1]+".txt")
time = float(sys.argv[2])
destination = str(sys.argv[3])

path = "replays\\"

with open(sourceFile) as f:
        for line in f:
                #replayName = line[9:]
                replayName = line[:-1]
                replayName = path + replayName
                print("ReplayName:"+replayName)
                #replay_parser.mainFunction(line, time, destination)
                replay_parser.mainFunction(replayName, time, destination)
print("Done")
