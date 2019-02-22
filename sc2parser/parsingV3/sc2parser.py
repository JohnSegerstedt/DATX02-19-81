import sys
import replay_parser
from pathlib import Path

sourceFile = str(sys.argv[1]+".txt")
time = float(sys.argv[2])
destination = str(sys.argv[3])

path = "new_replays\\"

with open(sourceFile) as f:
        counter = 0
        for line in f:
                counter += 1
                replayName = line[12:]
                replayName = replayName[:-1]
                replayName = path + replayName
                print("Replay:"+str(counter)+" - ReplayName:"+replayName)
                #replay_parser.mainFunction(line, time, destination)
                replay_parser.mainFunction(replayName, time, destination)
print("Done")
