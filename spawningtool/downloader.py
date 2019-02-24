import requests


# in order to run this file you need to have folder
# 'spawningtool'
# beside the git repository DATX02-19-81 folder

downloads = []

with open("games.txt") as f:
    for line in f:
        url = line.strip()
        downloadChar = url.find("/download/")
        gameID = url[:downloadChar]
        lastSlash = gameID.rfind("/")
        gameID = gameID[lastSlash+1:]
        downloads.append({"gameID":gameID, "url":url})

length = len(downloads)
for idx, download in enumerate(downloads):
    print("Downloading replay", download["url"], idx, "of", length, " :: ", round((idx/length * 100), 2), "%")
        
    r = requests.get(download["url"], allow_redirects=True)
    open("../../spawningtool/" + download["gameID"] + ".SC2Replay", "wb").write(r.content)