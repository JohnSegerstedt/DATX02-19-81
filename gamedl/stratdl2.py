import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
import json

files2 = []
buildOrderCSVsDirectory = "../../"
files = os.listdir(buildOrderCSVsDirectory)
for f in files:
    if (f.endswith(".csv")):
        files2.append(f)

        

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

gameStrats = []
download = False
createColumns = True

def getStrats():
    for page in range(1, pages):
        print("Requesting url", getUrl(page, ""))
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

                if (download):
                    print("Downloading " + gameID)
                    r = requests.get("https://lotv.spawningtool.com/" + gameID + "/download", allow_redirects=True)
                    open("../../BuildOrderReplays3/" + gameID + ".SC2Replay", "wb").write(r.content)

                gameStrats.append({"gameID":gameID, "P1":span1.text, "P2":span2.text})
    
    with (open("strats.json", "w")) as outfile:
        json.dump(gameStrats, outfile)
                

def assignStrats():
    print("Assigning strats")
    print(files2)
    myStrats = []
    with open("strats.json") as strats:
        myStrats = json.load(strats)

    for f in files2:
        print(f)
        df = pd.read_csv(buildOrderCSVsDirectory + f)
        if (createColumns):
            df["P1"] = "[]"
            df["P2"] = "[]"

        for strat in myStrats:
            try:
                df.loc[df["Name"].str.contains(strat["gameID"]+".SC2Replay"), "P1"] = "[" + strat["P1"] + "]"
                df.loc[df["Name"].str.contains(strat["gameID"]+".SC2Replay"), "P2"] = "[" + strat["P2"] + "]"
            except:
                print("Failed finding gameID: " + strat["gameID"] + " in file: " + f)
        
        df.to_csv(buildOrderCSVsDirectory + f, index=False)


print("Requesting url", getUrl(1, ""))
res = requests.get(getUrl(1, ""))
soup = BeautifulSoup(res.text, "html.parser")

pagesString = soup.findAll("h3")[1].text
ofIndex = pagesString.find("of")
pages = int(pagesString[ofIndex+3:-1])
print("pages:", pages)

#getStrats()
assignStrats()