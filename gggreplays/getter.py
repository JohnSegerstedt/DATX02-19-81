
import requests
import pprint
import json

pp = pprint.PrettyPrinter(indent=4)
limitParam = 10 #There are ~4200 matches of Platinum, ~2500 matches of Diamond, ~500 matches of Master
leagueParam = 4 #3 for Platinum, 4 for Diamond, 5 for Master
url = "https://gggreplays.com/api/v1/matches?average_league=" + str(leagueParam) + "&game_type=1v1&replay=true&vs_race=protoss&limit=" + str(limitParam)


response = requests.get(url)

myJson = json.loads(response.text)

# pp.pprint(myJson["collection"][0]["id"])

listLength = len(myJson["collection"])

for x in range(0, listLength):
    id = myJson["collection"][x]["id"]
    urlDl = "https://gggreplays.com/matches/" + str(id) + "/replay"
    print("Downloading replay ", urlDl, ", ", str(x), "of", listLength, " :: ", round((x/listLength * 100), 2), "%")
    r = requests.get(urlDl, allow_redirects=True)
    # Skapa denna mappen!
    open("../../gggreplays/" + str(id) + ".SC2Replay", "wb").write(r.content)





