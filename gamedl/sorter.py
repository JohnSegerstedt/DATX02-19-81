import sc2reader
import os

mainDir = "../../replays/" #The directory containing the replay files, change if needed.

def sort(targetPath):
    files = [f for f in os.listdir(targetPath) if os.path.isfile(os.path.join(targetPath, f))]
    print("--- Renaming and sorting", files.__len__(), "replays ---")
    sortedCount = 0
    badMatchCount = 0
    existingCount = 0

    for x in range(0, files.__len__()):
        file = files[x]

        if x % 10 == 0:
            print("Sorting replays... :: ", round((x/files.__len__() * 100), 2), "%")

        try:
            replay = sc2reader.load_replay(targetPath + file, load_level=2)

            #Skip matches with anything other than 2 players
            playerCount = 0
            for team in replay.teams:
                for player in team.players:
                    playerCount += 1
            if playerCount != 2:
                badMatchCount += 1
                print("Bad match found, does not contain exactly 2 players:", targetPath + file)
                continue

            #Skip non-PvP matches
            if replay.teams[0].lineup + replay.teams[1].lineup != "PP" and replay.teams[0].lineup + replay.teams[1].lineup != "ПП":
                badMatchCount += 1
                print("Bad match found, not a PvP game:", targetPath + file)
                continue

            #Skip matches with AI
            if replay.teams[0].players[0].name.startswith("A.I.") or replay.teams[1].players[0].name.startswith("A.I."):
                badMatchCount += 1
                print("Bad match found, contains an AI actor:", targetPath + file)
                continue

            #Build new file name
            version = replay.release_string.replace('.', '-')
            newFileName = version + "-" + replay.end_time.isoformat().replace(':', '-') + "-" + str(replay.game_length.seconds) + ".SC2Replay"
            subDir = "sorted/"

            if not os.path.isfile(targetPath + subDir + newFileName):
                if not os.path.isdir(targetPath + subDir):
                    os.makedirs(targetPath + subDir)
                sortedCount += 1
                os.rename(targetPath + file, targetPath + subDir + newFileName)
            else:
                existingCount += 1
                print("Replay already exists. Old file:", targetPath + file, ", new file:", targetPath + subDir + newFileName)
        except:
            badMatchCount += 1
            print("Bad match found, corrupt or missing data probably:", targetPath + file)
    print("--- Done! Sorted", files.__len__(), "replays.", sortedCount, "successful,", existingCount, "already existing,", badMatchCount, "bad matches found. ---")

def main():
    sort(mainDir)

if __name__ == "__main__":
   main()