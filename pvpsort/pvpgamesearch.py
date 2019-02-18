import sys
import os
import sc2reader

def main(): 
    fileLocation = "pvpgames.txt"
    # this line erases all previous text in the file
    open(fileLocation, "w").close()

    # loop through all games. check if both races are "P",
    # if "P", "P": 
    #       append to .txt file of PvP game replay locations (file names)
    myfile = open(fileLocation, "w")
    paths = "../../replays/3.16.1-Pack_1-fix/Replays/"
    replays = sc2reader.load_replays(paths, load_level=2)
    length = len(os.listdir(paths))

    pvpcount = 0
    for idx, replay in enumerate(replays):
        print("Checking replay", idx, " of ", length, " :: Progress ", round((idx/length * 100), 2), "% :: PvPs found: ", pvpcount)
        races = getRaces(replay)
        if (races[0] is "P" and races[1] is "P"): 
            myfile.write(replay.filename + "\n")
            pvpcount += 1

def getRaces(replay):
    p1race = None
    p2race = None
    for team in replay.teams:
        for player in team:
            if p1race is None:
                p1race = player.pick_race[0]
            else:
                p2race = player.pick_race[0]
    return [p1race, p2race]

if __name__ == '__main__':
    main()