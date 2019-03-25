import requests
from bs4 import BeautifulSoup
import os

orders = [  # Or rather build orders
    {"tag": "72", "name": "1GateExpand"},
    {"tag": "1977", "name": "ArchonImmortal"},
    {"tag": "20", "name": "BlinkStalkers"},
    {"tag": "53", "name": "DarkTemplarRush"},
    {"tag": "22", "name": "OracleOpening"},
    {"tag": "52", "name": "PheonixOpening"},
    {"tag": "80", "name": "ZealotArchon"}
]


def getUrl(page, tag):
    return "https://lotv.spawningtool.com/replays/?p=" + str(page) + "&pro_only=on&tag=" + str(tag) + "&tag=9&query=&after_time=&before_time=&after_played_on=&before_played_on=&coop=&patch=&order_by="


print("Requesting url", getUrl(1, ""))
res = requests.get(getUrl(1, ""))
soup = BeautifulSoup(res.text, "html.parser")

pagesString = soup.findAll("h3")[1].text
ofIndex = pagesString.find("of")
pages = int(pagesString[ofIndex+3:-1])
print("pages:", pages)


for page in range(1, pages):
    print("Requesting url", getUrl(1, ""))
    res = requests.get(getUrl(page, ""))
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
            gameUrl = "https://lotv.spawningtool.com/"+gameID

            res = requests.get(gameUrl)
            soup = BeautifulSoup(res.text, "html.parser")
            forms = soup.find_all(
                "form", {"class": "add-tag-form simple form-inline"})
            print("GameID: " + gameID)

            span1 = forms[0].find("span")
            span2 = forms[1].find("span")
            print("Player 1: " + span1.text)
            print("Player 2: " + span2.text)