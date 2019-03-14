import pymongo
import enum
import pprint
import pandas as pd

# Generates .CSV from connected MongoDB Server. 
# Choose frames to use, database name and filename. 
# The .csv is dynamically generated, employing UnitId and UpgradeId to match unit- and upgrade number respectively to their corresponding name. 
# Should any columns have strange names inspect these Enum classes first.

# INTERESTING VARIABLES
# Choose which frames to extract data for. Frames come in sizes 12*x
framesOfInterest = [11568, 1200]#,2400,3600]
databaseName = "vision_testing"
destinationFileName = "fileName"
printResult = True

# ----- SETUP -----
#extra = ["Replay_id", "Frame_id"]
#units = ["Adept","Archon","Carrier","Colossus","DarkTemplar","Disruptor","HighTemplar","Immortal","Mothership","Observer","Oracle","Phoenix","Probe","Sentry","Stalker","Tempest","WarpPrism","VoidRay","Zealot","Assimilator","CyberneticsCore","DarkShrine","FleetBeacon","Forge","Gateway","Nexus","PhotonCannon","Pylon","RoboticsFacility","RoboticsBay","ShieldBattery","Stargate","TemplarArchive","TwilightCouncil","WarpGate"]
#upgrades = ["ProtossAirArmorsLevel1","ProtossAirArmorsLevel2","ProtossAirArmorsLevel3","ProtossAirWeaponsLevel1","ProtossAirWeaponsLevel2","ProtossAirWeaponsLevel3","ProtossGroundArmorsLevel1","ProtossGroundArmorsLevel2","ProtossGroundArmorsLevel3","ProtossGroundWeaponsLevel1","ProtossGroundWeaponsLevel2","ProtossGroundWeaponsLevel3","ProtossShieldsLevel1","ProtossShieldsLevel2","ProtossShieldsLevel3","AdeptPiercingAttack","BlinkTech","Charge","DarkTemplarBlinkUpgrade","ExtendedThermalLance","GraviticDrive","ObserverGraviticBooster","PhoenixRangeUpgrade","PsiStormTech","WarpGateResearch"]

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

# EXTRACT UNIT, UPGRADE AND SEEN-UNIT DATA FOR BOTH PLAYERS IN A STATE
def extractData(state):
  global dictUpgradeIndex
  global dictUnitIndex
  
  stateData = {}
  stateData["0Replay_id"] = state["replay_id"]
  stateData["0Frame_id"] = state["frame_id"]

  #extracts unit count with player id appended, ie a probe belonging to player 1 will be called "Probe1"
  for unitKey in state["units"].keys():
    unitType = ("P" + str(state["player_id"]) + "_Unit_" + UnitId(int(unitKey)).name)
    stateData[unitType] = len(state["units"][unitKey])

  #extracts upgrades with player id appended, ie: "BlinkTech1"
  for upgrade in state["upgrades"]:
    upgradeType = ( "P" + str(state["player_id"]) + "_Upgrade_" + UpgradeId(int(upgrade)).name)
    stateData[upgradeType] = 1

  #Extracts units spotted by type. If player one has spotted a Probe it will be called "ProbeSpottedBy1"
  for unitKey in state["seen_enemy_units"].keys():
  	unitType = ("P" + str(state["player_id"]) + "_HasSpotted_" + UnitId(int(unitKey)).name)
  	stateData[unitType] = (state["seen_enemy_units"][unitKey]) #TODO light this one when unit names are fixed.

  return stateData

def saveToDataframe(data, playerId):
  global df_p1
  global df_p2
  
  if playerId == 1:
  	df_p1 = df_p1.append(data, ignore_index=True)
  else:
  	df_p2 = df_p2.append(data, ignore_index=True)

def saveAsCSV(dataFrame, fileName):
  dataFrame.to_csv("fileName" + ".csv")

def connectToDatabase(db):
  # Create mongodb client
  client = pymongo.MongoClient()
  # Connect to db using client
  db = client[db]
  return db

# Custom printing, for debugging purposes:
def printInfo(dataFrame):
	print("Columns containing only NaNs: ")
	res = merged.columns[merged.isnull().all(0)]
	print(res)
	print("Full dataframe print: ")
	with pd.option_context('display.max_rows', None, 'display.max_columns', None):
		print(dataFrame)
	print("Number of rows, columns: ", dataFrame.shape)


# ---- MAIN ----
db = connectToDatabase(databaseName)

# CREATE REFERENCES BY KEY
#actions = db["actions"]
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
df_p1 = pd.DataFrame(columns=["0Replay_id", "0Frame_id"]) #dtype='int'
df_p2 = pd.DataFrame(columns=["0Replay_id", "0Frame_id"]) #dtype='int'

# RETRIEVE STATE DATA FROM MONGO
stateData = db.states.aggregate([
  #get matching stuff
    { "$match" : {"frame_id" : {"$in" : framesOfInterest }, } },
  #define what to expose from the matching above.
    {"$project" : {"replay_id": 1, "_id" : 0, "frame_id" : 1, "player_id" : 1, "units" : 1, "upgrades" : 1, "seen_enemy_units" : 1 }},
    ])

# PARSE STATE DATA.
count = 0
for state in stateData:
  newRow = extractData(state)
  saveToDataframe(newRow, state["player_id"])
  count += 1

print("Finished, total states parsed:", count)

#Merge player 1 and player 2 data into one joint dataframe. 
merged = pd.merge(df_p1,df_p2, on=['0Replay_id', '0Frame_id'], how = "outer")

if printResult:
	printInfo(merged)
# Replace all NaN values with zero's
merged = merged.fillna(0)
# Sort by name
merged = merged.reindex(sorted(merged.columns), axis=1)
saveAsCSV(merged, destinationFileName)























# POTENTIALLY USEFUL LATER:

# RETRIEVE VISIBLE ENEMY UNITS DATA FROM MONGO. - In progress, coding halted as it may not be needed.
#replay_ids = []
#for replay_doc in replays.find({}, {"replay_id": 1}):
#	replay_ids.append(replay_doc["replay_id"])
#
#for replay_id in replay_ids:
#	stateData = db.states.aggregate([
#  #get matching stuff
#	    { "$match" : {"replay_id" : replay_id, "player_id" : 1} },
#	  #define what to expose from the matching above.
#	    {"$project" : {"replay_id": 1, "_id" : 0, "frame_id" : 1, "player_id" : 1, "visible_enemy_units" : 1 }}, 
#	    {"$sort" : { "frame_id" : 1} }
#	    ])
#
#	for state in stateData:
#		print("state: ", state["replay_id"], "frame: ", state["frame_id"])
#		seenUnits = extractSeenUnits(state)
#		print(seenUnits)

# EXTRACTS SEEN UNITS BASED ON visible_enemy_units - may not be needed, unfinished. 
#def extractSeenUnits(state):
#	stateData = {}
#
#	stateData["Replay_id"] = state["replay_id"]
#	stateData["Frame_id"] = state["frame_id"]
#	for unitKey in state["visible_enemy_units"].keys():
#		print()
#		unitType = UnitId(int(unitKey)).name + str(state["player_id"])
#		stateData[unitType] = len(state["visible_enemy_units"][unitKey])
#	    #print(unitType)
#	    #print(len(state["units"][unitKey]))
#	return stateData