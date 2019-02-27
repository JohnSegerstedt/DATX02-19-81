import os
import requests
import zipfile
import pprint
import json
import sc2reader
from bs4 import BeautifulSoup

targetPath = "../../replays/"

def main():
    dlGggreplays(3)
    dlGggreplays(4)
    dlGggreplays(5)
    betterDlSpawningToolReplays()
    sortReplays()

def betterDlSpawningToolReplays():
    pageTotal = 38  # There are 38 pages of PvP replays
    url = "https://lotv.spawningtool.com/zip/?query=&after_time=&patch=&pro_only=on&before_played_on=&coop=&before_time=&order_by=&after_played_on=&tag=9&p="

    for x in range(1, pageTotal + 1):
        print("Downloading page", x, "of", pageTotal, " :: ", round((x / pageTotal * 100), 2), "%")
        r = requests.get(url + str(x), allow_redirects=True)
        # Skapa denna mappen!
        open(targetPath + str(x) + ".zip", "wb").write(r.content)
        print("Unzipping page", x, "of", pageTotal)
        zip_ref = zipfile.ZipFile(targetPath + str(x) + ".zip", 'r')
        zip_ref.extractall(targetPath)
        zip_ref.close()
        os.remove(targetPath + str(x) + ".zip")

def dlSpawningToolReplays():
    page = 1

    print("Starting SpawningTool Downloading")
    print(getSpawningToolUrl(page))
    res = requests.get(getSpawningToolUrl(page))
    print(res)
    soup = BeautifulSoup(res.text, "html.parser")

    pagesString = soup.findAll("h3")[1].text
    ofIndex = pagesString.find("of")
    pages = int(pagesString[ofIndex+3:-1])

    downloads = []

    for x in range(1, pages):
        print("Getting page", x, "of", pages)
        url = getSpawningToolUrl(x)
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")

        hrefs = soup.findAll("a")
        for href in hrefs:
            if ("download" in href.get("href")):
                dlString = href.get("href")
                dlString = dlString.strip()
                downloadChar = dlString.find("/download/")
                gameID = dlString[:downloadChar]
                lastSlash = gameID.rfind("/")
                gameID = gameID[lastSlash+1:]
                downloads.append({"gameID":gameID, "url":dlString})

    for download in downloads:
        r = requests.get("https://spawningtool.com" + download["url"], allow_redirects=True)
        open(targetPath + "spawningtool" + download["gameID"] + ".SC2Replay", "wb").write(r.content)


def dlGggreplays(league):
    limitParam = 5000  # There are ~4200 matches of Platinum, ~2500 matches of Diamond, ~500 matches of Master
    leagueParam = league  # 3 for Platinum, 4 for Diamond, 5 for Master
    # Target path for replay files. Make sure the folder exists!

    url = "https://gggreplays.com/api/v1/matches?average_league=" + \
        str(leagueParam) + \
        "&game_type=1v1&replay=true&vs_race=protoss&limit=" + str(limitParam)

    print("Starting dlGggreplays")

    response = requests.get(url)

    myJson = json.loads(response.text)

    # pp.pprint(myJson["collection"][0]["id"])

    listLength = len(myJson["collection"])

    for x in range(0, listLength):
        id = myJson["collection"][x]["id"]
        if os.path.isfile(targetPath + "gggreplays" + str(id) + ".SC2Replay"):
            print("Skipped replay ", str(x), "of", listLength,
                  ", file already exists.", " :: ", round((x/listLength * 100), 2), "%")
            continue
        urlDl = "https://gggreplays.com/matches/" + str(id) + "/replay"
        print("Downloading replay ", urlDl, ", ", str(x), "of",
              listLength, " :: ", round((x/listLength * 100), 2), "%")
        r = requests.get(urlDl, allow_redirects=True)
        open(targetPath + "gggreplays" + str(id) + ".SC2Replay", "wb").write(r.content)


def getSpawningToolUrl(page):
    return "https://lotv.spawningtool.com/replays/?p=" + str(page) + "&pro_only=on&tag=9&query=&after_time=&before_time=&after_played_on=&before_played_on=&coop=&patch=&order_by="

def sortReplays():
    mainDir = targetPath
    files = [f for f in os.listdir(mainDir) if os.path.isfile(os.path.join(mainDir, f))]
    print("--- Renaming and sorting", files.__len__(), "replays ---")

    for x in range(0, files.__len__()):
        file = files[x]

        if x % 10 == 0:
            print("Sorting replays... :: ", round((x / files.__len__() * 100), 2), "%")

        replay = sc2reader.load_replay(mainDir + file, load_level=2)

        # Skip matches with anything other than 2 players
        playerCount = 0
        for team in replay.teams:
            for player in team.players:
                playerCount += 1
        if playerCount != 2:
            print("Bad match found, does not contain exactly 2 players:", mainDir + file)
            continue

        # Skip non-PvP matches
        if (replay.teams[0].players[0].pick_race + replay.teams[1].players[0].pick_race) != "ProtossProtoss":
            print("Bad match found, not a PvP game:", mainDir + file)
            continue

        # Skip matches with AI
        if replay.teams[0].players[0].name.startswith("A.I.") or replay.teams[1].players[0].name.startswith("A.I."):
            print("Bad match found, contains an AI actor:", mainDir + file)
            continue

        # Build new file name
        version = replay.release_string.replace('.', '-')
        newFileName = version + "-" + replay.end_time.isoformat().replace(':', '-') + "-" + str(
            replay.game_length.seconds) + ".SC2Replay"
        subDir = "sorted/"

        if not os.path.isfile(mainDir + subDir + newFileName):
            if not os.path.isdir(mainDir + subDir):
                os.makedirs(mainDir + subDir)
            os.rename(mainDir + file, mainDir + subDir + newFileName)
        else:
            print("Replay already exists. Old file:", mainDir + file, ", new file:", mainDir + subDir + newFileName)

if(__name__ == '__main__'):
    main()
