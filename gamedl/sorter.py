import sc2reader
import os
import traceback
from pprint import pprint

#--- Feel free to touch these ---#
mainDir = "../../replays/" #The directory containing the replay files, change if needed.

DELETE_BAD_MATCHES = False

SEMI_ACCEPTED_LEAGUE = 3
ACCEPTED_LEAGUE = 4

LOWEST_LENGTH = 60 #Lowest accepted game time in seconds
HIGHEST_LENGTH = 2000 #Highest accepted game time in seconds

LOWEST_PATCH = 421
#LOWEST_PATCH = 0
HIGHEST_PATCH = 462
#HIGHEST_PATCH = 999

#--- You probably don't need to touch these ---#
subDir = "sorted/"
semiAcceptedDir = "semi_accepted_sorted/"
badDir = "bad_matches/"

def sort(targetPath):
    files = [f for f in os.listdir(targetPath) if os.path.isfile(os.path.join(targetPath, f)) and f.lower().endswith(".sc2replay")]
    print("--- Renaming and sorting", files.__len__(), "replays ---")
    sortedCount = 0
    semiSortedCount = 0
    notPvpCount = 0
    lowLeagueCount = 0
    wrongPatchCount = 0
    otherBadCount = 0
    existingCount = 0

    for x in range(0, files.__len__()):
        file = files[x]

        if x % 20 == 0:
            print("Sorting replays... :: ", round((x/files.__len__() * 100), 2), "%")

        try:
            replay = sc2reader.load_replay(targetPath + file, load_level=2)
            #pprint(vars(replay.teams[0].players[0]))
            #print(replay.teams[0].players[0].highest_league, replay.teams[1].players[0].highest_league)
            #print(replay.teams[0].players[0].play_race, replay.teams[1].players[0].play_race)

            #continue
            #Skip matches that are too short or too long
            if replay.game_length.seconds < LOWEST_LENGTH or replay.game_length.seconds > HIGHEST_LENGTH:
                otherBadCount += 1
                sortBadMatch(file, targetPath, "badlength")
                print("Bad match found, game is too short or too long:", replay.game_length.seconds, targetPath + file)
                continue

            #Skip matches with anything other than 2 players
            playerCount = 0
            for team in replay.teams:
                for player in team.players:
                    playerCount += 1
            if playerCount != 2:
                otherBadCount += 1
                sortBadMatch(file, targetPath, "not1v1")
                print("Bad match found, does not contain exactly 2 players:", targetPath + file)
                continue

            #Skip low league matches
            league = min(replay.teams[0].players[0].highest_league, replay.teams[1].players[0].highest_league)
            if league < SEMI_ACCEPTED_LEAGUE:
                lowLeagueCount += 1
                sortBadMatch(file, targetPath, "lowleague-" + str(league))
                #print("Bad match found, too low league:", targetPath + file)
                continue

            #Skip non-PvP matches
            if replay.teams[0].lineup + replay.teams[1].lineup != "PP" and replay.teams[0].lineup + replay.teams[1].lineup != "ПП":
                notPvpCount += 1
                sortBadMatch(file, targetPath, "notpvp")
                #print("Bad match found, not a PvP game:", targetPath + file)
                continue

            #Skip matches with AI
            if replay.teams[0].players[0].is_human == False or replay.teams[1].players[0].is_human == False:
                otherBadCount += 1
                sortBadMatch(file, targetPath, "botgame")
                print("Bad match found, contains an AI actor:", targetPath + file)
                continue

            #Skip matches outside the accepted patch range
            patchint = int(replay.release_string[:5].replace('.', ''))
            if patchint < LOWEST_PATCH or patchint > HIGHEST_PATCH:
                wrongPatchCount += 1
                sortBadMatch(file, targetPath, "badversion-" + str(patchint))
                print("Bad match found, wrong version:", targetPath + file)
                continue

            #Build new file name
            version = replay.release_string.replace('.', '-')
            newFileName = version + "-" + str(league) + '-' + replay.end_time.isoformat().replace(':', '-') + "-" + str(replay.game_length.seconds) + ".SC2Replay"


            if league < ACCEPTED_LEAGUE:
                if not os.path.isfile(targetPath + semiAcceptedDir + newFileName):
                    semiSortedCount += 1
                    sortSemiGoodMatch(file, targetPath, newFileName)
                else:
                    existingCount += 1
                    sortBadMatch(file, targetPath, "existing")
                    print("Replay already exists. Old file:", targetPath + file, ", new file:", targetPath + semiAcceptedDir + newFileName)
            else:
                if not os.path.isfile(targetPath + subDir + newFileName):
                    sortedCount += 1
                    sortGoodMatch(file, targetPath, newFileName)
                else:
                    existingCount += 1
                    sortBadMatch(file, targetPath, "existing")
                    print("Replay already exists. Old file:", targetPath + file, ", new file:", targetPath + subDir + newFileName)
        except Exception as e:
            otherBadCount += 1
            sortBadMatch(file, targetPath, "corrupt")
            print("Bad match found, corrupt or missing data probably:", targetPath + file)
            print(e)
            traceback.print_exc()
    print("--- Done! Sorted", files.__len__(), "replays.", sortedCount, "successful,", semiSortedCount, "semi-successful,", existingCount, "already existing,", notPvpCount, "non-pvp,", lowLeagueCount, "low league,", wrongPatchCount, "wrong patch,", otherBadCount, "other bad matches. ---")

def sortBadMatch(file, targetPath, errorString):
    if DELETE_BAD_MATCHES:
        os.remove(targetPath + file)
    else:
        if not os.path.isdir(targetPath + badDir):
            os.makedirs(targetPath + badDir)
        os.rename(targetPath + file, targetPath + badDir + errorString + "-" + file)

def sortGoodMatch(file, targetPath, newFileName):
    if not os.path.isdir(targetPath + subDir):
        os.makedirs(targetPath + subDir)
    os.rename(targetPath + file, targetPath + subDir + newFileName)

def sortSemiGoodMatch(file, targetPath, newFileName):
    if not os.path.isdir(targetPath + semiAcceptedDir):
        os.makedirs(targetPath + semiAcceptedDir)
    os.rename(targetPath + file, targetPath + semiAcceptedDir + newFileName)

def main():
    sort(mainDir)

if __name__ == "__main__":
   main()
