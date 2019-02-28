import pymongo
import enum
#import pandas as pd

#more lists can be found here: https://github.com/deepmind/pysc2/tree/master/pysc2/lib
class Protoss(enum.IntEnum):
  Adept = 311
  AdeptPhaseShift = 801
  Archon = 141
  Assimilator = 61
  Carrier = 79
  Colossus = 4
  CyberneticsCore = 72
  DarkShrine = 69
  DarkTemplar = 76
  Disruptor = 694
  DisruptorPhased = 733
  FleetBeacon = 64
  ForceField = 135
  Forge = 63
  Gateway = 62
  HighTemplar = 75
  Immortal = 83
  Interceptor = 85
  Mothership = 10
  MothershipCore = 488
  Nexus = 59
  Observer = 82
  ObserverSurveillanceMode = 1911
  Oracle = 495
  Phoenix = 78
  PhotonCannon = 66
  Probe = 84
  Pylon = 60
  PylonOvercharged = 894
  RoboticsBay = 70
  RoboticsFacility = 71
  Sentry = 77
  ShieldBattery = 1910
  Stalker = 74
  Stargate = 67
  StasisTrap = 732
  Tempest = 496
  TemplarArchive = 68
  TwilightCouncil = 65
  VoidRay = 80
  WarpGate = 133
  WarpPrism = 81
  WarpPrismPhasing = 136
  Zealot = 73



# Create mongodb client
client = pymongo.MongoClient()

# Connect to db using client
db = client["replay_database_test"]

# Create references by key
actions = db["actions"]
players = db["players"]
replays = db["replays"]
scores = db["scores"]
states = db["states"]

# Creating indices to reduce lookup time
replays.create_index("replay_id")
players.create_index("replay_id")
states.create_index([("replay_id", pymongo.ASCENDING), ("frame_id", pymongo.ASCENDING)])
actions.create_index([("replay_id", pymongo.ASCENDING), ("frame_id", pymongo.ASCENDING)])
scores.create_index([("replay_id", pymongo.ASCENDING), ("frame_id", pymongo.ASCENDING)])

# For each document in search result add result to replay_ids
replay_ids = []
for replay_doc in replays.find({}, {"replay_id": 1}):
	replay_ids.append(replay_doc["replay_id"])

for a in replay_ids:
	print(a + "\n")

# TODO Count spotted units for one game
# loop through each frame gathering build
# states_visible_enemy_units = []
#replay_1 = replay_ids[-1]
#for state in states.find({"replay_id": replay_1, "player_id": 1}).sort("frame_id", 1):
#	print(f'frame: {state["frame_id"]}, supply: {state["supply"]}')

# TODO For each game,
	#For frame x in game y get state
#Hardcoded attempt: 
p1_units = []
p2_units = []
replay_1 = replay_ids[0]
#for state in states.find({"replay_id": replay_1, "player_id": 1, "frame_id":2400}):
for state in states.find({"replay_name": 'C:\\Users\\da_fo\\Desktop\\kandidat\\sc2reader\\ggtracker_231557.SC2Replay', "player_id": 1, "frame_id":2400}):
	print(state["_id"])
	print(f'frame: {state["frame_id"]}, {state["player_id"]}, supply: {state["supply"]}')
	print(f'frame: {state["frame_id"]}, {state["player_id"]}, units: {len(state["units"])} ')
	for key in state["units"].keys():
		print(Protoss(int(key)).name)
		print(len(state["units"][key]))
	#len(dict)


# TODO save as csv
# df = pd.DataFrame([list(players[0].entities.values()) + list(players[1].entities.values())], index=index, columns = list(players[0].entities.keys()) + p1keylist)
# fileName = str(destination)+"-"+str(maxRealSeconds)+"s.csv"
# df.to_csv(fileName)



