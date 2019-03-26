import pymongo
import enum
import pprint
import pandas as pd
import os

# Generates .CSV from connected MongoDB Server. 
# Choose frames to use and database name. 
# The .csv is dynamically generated, employing UnitId and UpgradeId to match unit- and upgrade number respectively to their corresponding name. 
# Should any columns have strange names inspect these Enum classes first.

# INTERESTING VARIABLES
# Choose which frame_id's to extract data from. 
# Every 120 frame_id's contains unit data
framesOfInterest = list(range(5040, 21600, 720))
#framesOfInterest = [2880]
databaseName = "reaping2"
printResult = False
printResultShort = False

parseFlag = 1 # 0 = all data, 1 = cluster data, 2 = training data

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
  global parseFlag
 
  playerId = state["player_id"]

  stateData = {}
  stateData["0Replay_id"] = state["replay_id"]
  stateData["0Frame_id"] = state["frame_id"]

  #extracts unit count with player id, ie a probe belonging to player 1 will be called "P1_Unit_probe"
  for unitKey in state["units"].keys():
    unitType = ("P" + str(playerId) + "_Unit_" + UnitId(int(unitKey)).name)
    stateData[unitType] = len(state["units"][unitKey])

  #extracts upgrades with player id appended, ie: "P1_Upgrade_blink"
  for upgrade in state["upgrades"]:
    upgradeType = ( "P" + str(playerId) + "_Upgrade_" + UpgradeId(int(upgrade)).name)
    stateData[upgradeType] = 1

  #Extracts units spotted by type. If player one has spotted a Probe it will be called "P1_HasSpotted_nexus"
  if parseFlag == 0 or parseFlag == 2 :
  	for unitKey in state["seen_enemy_units"].keys():
  		unitType = ("P" + str(playerId) + "_HasSpotted_" + UnitId(int(unitKey)).name)
  		stateData[unitType] = (state["seen_enemy_units"][unitKey]) 

  # Extract minerals and gas
  for resource in state["resources"].keys():
  	columnName = ("P" + str(playerId) + "_" + resource)
  	stateData[columnName] = (state["resources"][resource]) 

  # Extract supply; {used, total, army, workers}
  for supplyType in state["supply"].keys():
  	columnName = ("P" + str(playerId) + "_Supply_" + supplyType)
  	stateData[columnName] = (state["supply"][supplyType])

  return stateData

def saveAsCSV(dataFrame, appendName, playerId):
  global parseFlag
  fileName = ""
  playerString = ""
  if parseFlag == 0:
  	fileName = "all_data"
  elif parseFlag == 1:
  	fileName = "cluster_data"
  elif parseFlag == 2:
  	fileName = "training_data"
  	playerString = "-" + str(playerId)
  dataFrame.to_csv("csvs/" + str(fileName) + str(appendName) + str(playerString) + ".csv")


def getMmr(replayId, playerId, database):
	playerData = database.players.aggregate([
	#get matching stuff
	{ "$match" : {"replay_id" : str(replayId), "player_id" : playerId} },
	#define what to expose from the matching above.
    {"$project" : {"player_mmr" : 1, "player_id" : 1}},
    ])

	for match in playerData:
		return match["player_mmr"]


def connectToDatabase(db):
  # Create mongodb client
  client = pymongo.MongoClient()
  # Connect to db using client
  db = client[db]
  return db

# Custom printing, for debugging purposes:
def printInfo(dataFrame):
	print("Columns containing only NaNs: ")
	res = dataFrame.columns[dataFrame.isnull().all(0)]
	print(res)
	print("Full dataframe print: ")
	with pd.option_context('display.max_rows', None, 'display.max_columns', None):
		print(dataFrame)
	print("Number of rows, columns: ", dataFrame.shape)

def printInfoShort(dataFrame):
	print("Columns containing only NaNs: ")
	res = dataFrame.columns[dataFrame.isnull().all(0)]
	print(res)
	print("Number of rows, columns: ", dataFrame.shape)

def extractFrameState(frameId):
	# INITIATING DATA FRAMES
	df_p1 = pd.DataFrame(columns=["0Replay_id", "0Frame_id"]) #dtype='int'
	df_p2 = pd.DataFrame(columns=["0Replay_id", "0Frame_id"]) #dtype='int'
	stateData = db.states.aggregate([
		  #get matching stuff
		    #{ "$match" : {"frame_id" : {"$in" : framesOfInterest }, } },
		    { "$match" : {"frame_id" : frameId} },
		  #define what to expose from the matching above.
		    {"$project" : {"resources" : 1, "replay_id": 1, "_id" : 0, "frame_id" : 1, "player_id" : 1, "supply" : 1, "units" : 1, "upgrades" : 1, "seen_enemy_units" : 1 }},
		    ])

	# PARSE STATE DATA.
	count = 0

	for state in stateData:
	  newRow = extractData(state)
	  #get mmr
	  mmr = getMmr(state["replay_id"], state["player_id"], db)
	  newRow["0P" + str(state["player_id"]) + "_mmr"] = mmr 
	  #saveToDataframe(newRow, state["player_id"], df_p1, df_p2)
	  if state["player_id"] == 1:
	  	df_p1 = df_p1.append(newRow, ignore_index=True)
	  elif state["player_id"] == 2:
	  	df_p2 = df_p2.append(newRow, ignore_index=True)
	  count += 1
	  if (count%20==0): 
	  	print(count)
	  #if count >= 100:
	  #	break

	print("Finished " + str(frameId) +", total states parsed:", count)

	if parseFlag == 2:
		df_p1 = df_p1.fillna(0)
		df_p1 = df_p1.reindex(sorted(df_p1.columns), axis=1)
		df_p2 = df_p2.fillna(0)
		df_p2 = df_p2.reindex(sorted(df_p2.columns), axis=1)
		saveAsCSV(df_p1, frameId, 1)
		saveAsCSV(df_p2, frameId, 2)
	else: 
		#Merge player 1 and player 2 data into one joint dataframe. 
		merged = pd.merge(df_p1,df_p2, on=['0Replay_id', '0Frame_id'], how = "outer")

		if printResult:
			printInfo(merged)
		if printResultShort:
			printInfoShort(merged)
		# Replace all NaN values with zero's
		merged = merged.fillna(0)
		# Sort by name
		merged = merged.reindex(sorted(merged.columns), axis=1)
		
		saveAsCSV(merged, frameId, 0)


# ---- MAIN ----
db = connectToDatabase(databaseName)

# CREATE REFERENCES BY KEY
#actions = db["actions"]
players = db["players"]
replays = db["replays"]
#scores = db["scores"]
states = db["states"]

# CREATING INDICES TO REDUCE LOOKUP TIME
replays.create_index("replay_id")
players.create_index("replay_id")
states.create_index([("replay_id", pymongo.ASCENDING), ("frame_id", pymongo.ASCENDING)])
#actions.create_index([("replay_id", pymongo.ASCENDING), ("frame_id", pymongo.ASCENDING)])
#scores.create_index([("replay_id", pymongo.ASCENDING), ("frame_id", pymongo.ASCENDING)])

if not os.path.exists("csvs/"):
    os.makedirs("csvs/")

for frame in framesOfInterest:
	print("extracting frame" + str(frame)) 
	extractFrameState(frame)






















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
