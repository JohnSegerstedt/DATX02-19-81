import sys
from shutil import copyfile

def main():
    txtLocation = "pvpgames.txt"
    replayDestination = "../../replays/3.16.1-Pack_1-fix/PvP/"

    with open(txtLocation, "r") as f:
        for line in f:
            line1 = line.strip()
            gameStart = line.rfind("/") + 1
            line2 = line1[gameStart:]
            copyfile(line1, replayDestination+line2)

if __name__ == '__main__':
    main()