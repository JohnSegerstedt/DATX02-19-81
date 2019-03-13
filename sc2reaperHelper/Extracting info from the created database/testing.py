import pymongo
import enum
import pprint
import pandas as pd
import datetime

#Generates .CSV from connected MongoDB Server.

# INTERESTING VARIABLES
# Choose which frames to extract data for. Frames come in sizes 12*x
framesOfInterest = [11568] #1200,2400,3600]
databaseName = "vision_testing"

#Full game state: replay_id + frame_id + units1 + upgrades1 + units2 + upgrades2
#One player game state: replay_id + frame_id + units1 + upgrades1 + (for each unit type; #max spotted in a single frame up to the current frame) 

# ----- SETUP -----
extra = ["Replay_id", "Frame_id"]
units = ["Adept","Archon","Carrier","Colossus","DarkTemplar","Disruptor","HighTemplar","Immortal","Mothership","Observer","Oracle","Phoenix","Probe","Sentry","Stalker","Tempest","WarpPrism","VoidRay","Zealot","Assimilator","CyberneticsCore","DarkShrine","FleetBeacon","Forge","Gateway","Nexus","PhotonCannon","Pylon","RoboticsFacility","RoboticsBay","ShieldBattery","Stargate","TemplarArchive","TwilightCouncil","WarpGate"]
upgrades = ["ProtossAirArmorsLevel1","ProtossAirArmorsLevel2","ProtossAirArmorsLevel3","ProtossAirWeaponsLevel1","ProtossAirWeaponsLevel2","ProtossAirWeaponsLevel3","ProtossGroundArmorsLevel1","ProtossGroundArmorsLevel2","ProtossGroundArmorsLevel3","ProtossGroundWeaponsLevel1","ProtossGroundWeaponsLevel2","ProtossGroundWeaponsLevel3","ProtossShieldsLevel1","ProtossShieldsLevel2","ProtossShieldsLevel3","AdeptPiercingAttack","BlinkTech","Charge","DarkTemplarBlinkUpgrade","ExtendedThermalLance","GraviticDrive","ObserverGraviticBooster","PhoenixRangeUpgrade","PsiStormTech","WarpGateResearch"]

class UpgradeId(enum.Enum):
  ProtossAirArmorsLevel1 = 81
  ProtossAirArmorsLevel2 = 82
  ProtossAirArmorsLevel3 = 83
  ProtossAirWeaponsLevel1 = 78
  ProtossAirWeaponsLevel2 = 79
  ProtossAirWeaponsLevel3 = 80
  ProtossGroundArmorsLevel1 = 42
  ProtossGroundArmorsLevel2 = 43
  ProtossGroundArmorsLevel3 = 44
  ProtossGroundWeaponsLevel1 = 39
  ProtossGroundWeaponsLevel2 = 40
  ProtossGroundWeaponsLevel3 = 41
  ProtossShieldsLevel1 = 45
  ProtossShieldsLevel2 = 46
  ProtossShieldsLevel3 = 47
  AdeptPiercingAttack = 130
  BlinkTech = 87
  Charge = 86
  DarkTemplarBlinkUpgrade = 141
  ExtendedThermalLance = 50
  GraviticDrive = 49
  ObserverGraviticBooster = 48
  PhoenixRangeUpgrade = 99
  PsiStormTech = 52
  WarpGateResearch = 84
  #extra
  HALTECH = 85
  HIGHTEMPLARKHAYDARINAMULET = 51
  CARRIERLAUNCHSPEEDUPGRADE = 1
  ANIONPULSECRYSTALS = 112
  TEMPESTRANGEUPGRADE = 100
  ORACLEENERGYUPGRADE = 104
  RESTORESHIELDS = 105
  PROTOSSHEROSHIPWEAPON = 106
  PROTOSSHEROSHIPDETECTOR = 107
  PROTOSSHEROSHIPSPELL = 108
  IMMORTALREVIVE = 121
  ADEPTSHIELDUPGRADE = 126
  IMMORTALBARRIER = 128
  ADEPTKILLBOUNCE = 129
  ADEPTPIERCINGATTACK = 130
  COMBATSHIELD = 234
  SINGULARITYCHARGE = 256
  DARKPROTOSS = 260
  VOIDRAYSPEEDUPGRADE = 288
  CARRIERCARRIERCAPACITY = 294
  CARRIERLEASHRANGEUPGRADE = 295

#more lists can be found here: https://github.com/deepmind/pysc2/tree/master/pysc2/lib
class UnitId(enum.IntEnum):
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

def extractSeenUnits(state):
	stateData = {}

	stateData["Replay_id"] = state["replay_id"]
	stateData["Frame_id"] = state["frame_id"]
	for unitKey in state["visible_enemy_units"].keys():
		print()
		unitType = UnitId(int(unitKey)).name + str(state["player_id"])
		stateData[unitType] = len(state["visible_enemy_units"][unitKey])
	    #print(unitType)
	    #print(len(state["units"][unitKey]))
	return stateData

def extractData(state):
  global dictUpgradeIndex
  global dictUnitIndex
  
  stateData = {}
  stateData["Replay_id"] = state["replay_id"]
  stateData["Frame_id"] = state["frame_id"]

  for unitKey in state["units"].keys():
    unitType = UnitId(int(unitKey)).name + str(state["player_id"])
    #print(unitType)
    #print(len(state["units"][unitKey]))
    stateData[unitType] = len(state["units"][unitKey])

  for upgrade in state["upgrades"]:
    upgradeType = (UpgradeId(int(upgrade)).name + str(state["player_id"]))
    #print(upgradeType)
    stateData[upgradeType] = 1

  return stateData

def saveToDataframe(data, playerId):
  global df_p1
  global df_p2
  global df
  df = df.append(data, ignore_index=True)

  if playerId == 1:
  	print(playerId)
  	df_p1 = df_p1.append(data, ignore_index=True)
  else:
  	print(playerId)
  	df_p2 = df_p2.append(data, ignore_index=True)

def printCSV(dataFrame):
  #file = str(destination)
  global fileName
  dataFrame.to_csv("fileName" + ".csv")

def connectToDatabase(db):
  # Create mongodb client
  client = pymongo.MongoClient()
  # Connect to db using client
  db = client[db]
  return db

# ---- MAIN ----
db = connectToDatabase(databaseName)

# Create references by key
actions = db["actions"]
#players = db["players"]
replays = db["replays"]
#scores = db["scores"]
states = db["states"]

# CREATING INDICES TO REDUCE LOOKUP TIME
replays.create_index("replay_id")
#players.create_index("replay_id")
states.create_index([("replay_id", pymongo.ASCENDING), ("frame_id", pymongo.ASCENDING)])
#actions.create_index([("replay_id", pymongo.ASCENDING), ("frame_id", pymongo.ASCENDING)])
#scores.create_index([("replay_id", pymongo.ASCENDING), ("frame_id", pymongo.ASCENDING)])

# INITIATING DATA FRAMES
p1_units_upgrades = [entry + "1" for entry in (units + upgrades)]
p2_units_upgrades = [entry + "2" for entry in (units + upgrades)]
df_p1 = pd.DataFrame(columns=extra + p1_units_upgrades) #dtype='int'
df_p2 = pd.DataFrame(columns=extra + p2_units_upgrades)
df = pd.DataFrame(columns=extra + p1_units_upgrades + p2_units_upgrades)


# TODO generalize for all played games, and for each state of interest for each player
  # TODO iterate through each frame, count spotted units of each type, keeping track of max so far, for one game up until state of interest
  # states_visible_enemy_units = []


# RETRIEVE STATE DATA FROM MONGO
stateData = db.states.aggregate([
  #get matching stuff
    { "$match" : {"frame_id" : {"$in" : framesOfInterest }, } },
  #define what to expose from the matching above.
    {"$project" : {"replay_id": 1, "_id" : 0, "frame_id" : 1, "player_id" : 1, "units" : 1, "upgrades" : 1 }},
    ])

# PARSE STATE DATA
count = 0
for state in stateData:
  newRow = extractData(state)
  saveToDataframe(newRow, state["player_id"])
  count += 1
  #print("States saved:", count)

# RETRIEVE VISIBLE ENEMY UNITS DATA FROM MONGO
replay_ids = []
for replay_doc in replays.find({}, {"replay_id": 1}):
	replay_ids.append(replay_doc["replay_id"])

for replay_id in replay_ids:
	stateData = db.states.aggregate([
  #get matching stuff
	    { "$match" : {"replay_id" : replay_id, "player_id" : 1} },
	  #define what to expose from the matching above.
	    {"$project" : {"replay_id": 1, "_id" : 0, "frame_id" : 1, "player_id" : 1, "visible_enemy_units" : 1 }},
	    {"$sort" : { "frame_id" : 1} }
	    ])

	for state in stateData:
		print("state: ", state["replay_id"], "frame: ", state["frame_id"])
		seenUnits = extractSeenUnits(state)
		print(seenUnits)



# PARSE VISIBLE ENEMY UNITS DATA
#for replay_doc in state.find({}, {"replay_id": 1}):
#	print(replay_doc)

print("Finished, total states parsed:", count)

print("FINAL OUTPUT test")
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#    print(df_test)



pprint.pprint(df)


#TODO only one players stuff on each currently? Make two separate and merge p1 p2 like: 
print("DFP!")
print(df_p1)
print(df_p2)
#print(df_p1.columns.difference(df_p2.columns))

#Merge player 1 and player 2 data into one state. 
print(pd.merge(df_p1,df_p2, on=['Replay_id', 'Frame_id'], how = "outer"))
#https://pythonprogramming.net/join-merge-data-analysis-python-pandas-tutorial/

#TODO comopare columns. Rename to make them fit?
#TODO remove df throughout program
#TODO remove NaNs
printCSV(df)