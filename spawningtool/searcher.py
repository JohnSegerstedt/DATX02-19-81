import sys
from bs4 import BeautifulSoup
import requests
import os

page = 1

def getUrl(page):
    return "https://lotv.spawningtool.com/replays/?p=" + str(page) + "&pro_only=on&tag=9&query=&after_time=&before_time=&after_played_on=&before_played_on=&coop=&patch=&order_by="

res = requests.get(getUrl(page))
soup = BeautifulSoup(res.text, "html.parser")

pagesString = soup.findAll("h3")[1].text
ofIndex = pagesString.find("of")
pages = int(pagesString[ofIndex+3:-1])

downloads = []

for x in range(1, pages):
    print("Getting page", x, "of", pages)
    url = getUrl(x)
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")

    hrefs = soup.findAll("a")
    for href in hrefs:
        if ("download" in href.get("href")):
            downloads.append(href.get("href"))

gameFile = open("games.txt", "w")

for download in downloads:
    gameFile.write("https://lotv.spawningtool.com" + download + "\n")
gameFile.close()