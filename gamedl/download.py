import os
import requests
import pprint
import json
from bs4 import BeautifulSoup

targetPath = "../../replays/"

def main():
    dlGggreplays(3)
    dlGggreplays(4)
    dlGggreplays(5)
    dlSpawningToolReplays()

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


if(__name__ == '__main__'):
    main()
