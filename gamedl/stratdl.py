import requests
from bs4 import BeautifulSoup
import os

# Downloads and sorts games by specific strategy (build order) as given
# by spawningtools.com

targetPath = "../../replays/"

orders = [ # Or rather build orders 
    {"tag":"72", "name":"1GateExpand"}, 
    {"tag":"1977", "name":"ArchonImmortal"}, 
    {"tag":"20", "name":"BlinkStalkers"}, 
    {"tag":"53", "name":"DarkTemplarRush"},
    {"tag":"22", "name":"OracleOpening"},
    {"tag":"52", "name":"PheonixOpening"},
    {"tag":"80", "name":"ZealotArchon"}
]
def getUrl(page, tag):
    return "https://lotv.spawningtool.com/replays/?p=" + str(page) + "&pro_only=on&tag=" + str(tag) + "&tag=9&query=&after_time=&before_time=&after_played_on=&before_played_on=&coop=&patch=&order_by="

for order in orders:
    print("Requesting url", getUrl(1, order["tag"]))
    res = requests.get(getUrl(1, order["tag"]))
    soup = BeautifulSoup(res.text, "html.parser")

    pagesString = soup.findAll("h3")[1].text
    ofIndex = pagesString.find("of")
    pages = int(pagesString[ofIndex+3:-1])

    downloads = []

    for x in range(1, pages):
        print("Getting page", x, "of", pages)
        url = getUrl(x, order["tag"])
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

    directory = targetPath + "/" + order["name"] + "/"

    if (not os.path.exists(directory)):
         os.makedirs(directory)


    for download in downloads:
        print("Downloading", download["url"])
        r = requests.get("https://lotv.spawningtool.com" + download["url"], allow_redirects=True)
        open(directory + order["name"] + download["gameID"] + ".SC2Replay", "wb").write(r.content)

