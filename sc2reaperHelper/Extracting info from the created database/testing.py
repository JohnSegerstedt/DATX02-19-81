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
framesOfInterest = list(range(0, 21600, 720))
#framesOfInterest = [0]
databaseName = "reaping2"
printResult = False
printResultShort = False

parseFlag = 1 # 0 = all data, 1 = cluster data, 2 = training data

allColumns = ['0Frame_id', '0P1_mmr', '0P1_result', '0P2_mmr','0P2_result', '0Replay_id', 'P1_Supply_army', 'P1_Supply_total', 'P1_Supply_used', 'P1_Supply_workers', 'P1_Unit_Adept', 'P1_Unit_AdeptPhaseShift', 'P1_Unit_Archon', 'P1_Unit_Assimilator', 'P1_Unit_Carrier', 'P1_Unit_Colossus', 'P1_Unit_CyberneticsCore', 'P1_Unit_DarkShrine', 'P1_Unit_DarkTemplar', 'P1_Unit_Disruptor', 'P1_Unit_DisruptorPhased', 'P1_Unit_FleetBeacon', 'P1_Unit_Forge', 'P1_Unit_Gateway', 'P1_Unit_HighTemplar', 'P1_Unit_Immortal', 'P1_Unit_Interceptor', 'P1_Unit_Mothership', 'P1_Unit_Nexus', 'P1_Unit_Observer', 'P1_Unit_ObserverSurveillanceMode', 'P1_Unit_Oracle', 'P1_Unit_Phoenix', 'P1_Unit_PhotonCannon', 'P1_Unit_Probe', 'P1_Unit_Pylon', 'P1_Unit_RoboticsBay', 'P1_Unit_RoboticsFacility', 'P1_Unit_Sentry', 'P1_Unit_ShieldBattery', 'P1_Unit_Stalker', 'P1_Unit_Stargate', 'P1_Unit_StasisTrap', 'P1_Unit_Tempest', 'P1_Unit_TemplarArchive', 'P1_Unit_TwilightCouncil', 'P1_Unit_VoidRay', 'P1_Unit_WarpGate', 'P1_Unit_WarpPrism', 'P1_Unit_WarpPrismPhasing', 'P1_Unit_Zealot', 'P1_Upgrade_AdeptPiercingAttack', 'P1_Upgrade_BlinkTech', 'P1_Upgrade_CARRIERLAUNCHSPEEDUPGRADE', 'P1_Upgrade_Charge', 'P1_Upgrade_DarkTemplarBlinkUpgrade', 'P1_Upgrade_ExtendedThermalLance', 'P1_Upgrade_GraviticDrive', 'P1_Upgrade_ObserverGraviticBooster', 'P1_Upgrade_PhoenixRangeUpgrade', 'P1_Upgrade_ProtossAirArmorsLevel1', 'P1_Upgrade_ProtossAirArmorsLevel2', 'P1_Upgrade_ProtossAirArmorsLevel3', 'P1_Upgrade_ProtossAirWeaponsLevel1', 'P1_Upgrade_ProtossAirWeaponsLevel2', 'P1_Upgrade_ProtossAirWeaponsLevel3', 'P1_Upgrade_ProtossGroundArmorsLevel1', 'P1_Upgrade_ProtossGroundArmorsLevel2', 'P1_Upgrade_ProtossGroundArmorsLevel3', 'P1_Upgrade_ProtossGroundWeaponsLevel1', 'P1_Upgrade_ProtossGroundWeaponsLevel2', 'P1_Upgrade_ProtossGroundWeaponsLevel3', 'P1_Upgrade_ProtossShieldsLevel1', 'P1_Upgrade_ProtossShieldsLevel2', 'P1_Upgrade_ProtossShieldsLevel3', 'P1_Upgrade_PsiStormTech', 'P1_Upgrade_WarpGateResearch', 'P1_minerals', 'P1_vespene', 'P2_Supply_army', 'P2_Supply_total', 'P2_Supply_used', 'P2_Supply_workers', 'P2_Unit_Adept', 'P2_Unit_AdeptPhaseShift', 'P2_Unit_Archon', 'P2_Unit_Assimilator', 'P2_Unit_Carrier', 'P2_Unit_Colossus', 'P2_Unit_CyberneticsCore', 'P2_Unit_DarkShrine', 'P2_Unit_DarkTemplar', 'P2_Unit_Disruptor', 'P2_Unit_DisruptorPhased', 'P2_Unit_FleetBeacon', 'P2_Unit_Forge', 'P2_Unit_Gateway', 'P2_Unit_HighTemplar', 'P2_Unit_Immortal', 'P2_Unit_Interceptor', 'P2_Unit_Mothership', 'P2_Unit_Nexus', 'P2_Unit_Observer', 'P2_Unit_ObserverSurveillanceMode', 'P2_Unit_Oracle', 'P2_Unit_Phoenix', 'P2_Unit_PhotonCannon', 'P2_Unit_Probe', 'P2_Unit_Pylon', 'P2_Unit_RoboticsBay', 'P2_Unit_RoboticsFacility', 'P2_Unit_Sentry', 'P2_Unit_ShieldBattery', 'P2_Unit_Stalker', 'P2_Unit_Stargate', 'P2_Unit_StasisTrap', 'P2_Unit_Tempest', 'P2_Unit_TemplarArchive', 'P2_Unit_TwilightCouncil', 'P2_Unit_VoidRay', 'P2_Unit_WarpGate', 'P2_Unit_WarpPrism', 'P2_Unit_WarpPrismPhasing', 'P2_Unit_Zealot', 'P2_Upgrade_AdeptPiercingAttack', 'P2_Upgrade_BlinkTech', 'P2_Upgrade_CARRIERLAUNCHSPEEDUPGRADE', 'P2_Upgrade_Charge', 'P2_Upgrade_DarkTemplarBlinkUpgrade', 'P2_Upgrade_ExtendedThermalLance', 'P2_Upgrade_GraviticDrive', 'P2_Upgrade_ObserverGraviticBooster', 'P2_Upgrade_PhoenixRangeUpgrade', 'P2_Upgrade_ProtossAirArmorsLevel1', 'P2_Upgrade_ProtossAirArmorsLevel2', 'P2_Upgrade_ProtossAirArmorsLevel3', 'P2_Upgrade_ProtossAirWeaponsLevel1', 'P2_Upgrade_ProtossAirWeaponsLevel2', 'P2_Upgrade_ProtossAirWeaponsLevel3', 'P2_Upgrade_ProtossGroundArmorsLevel1', 'P2_Upgrade_ProtossGroundArmorsLevel2', 'P2_Upgrade_ProtossGroundArmorsLevel3', 'P2_Upgrade_ProtossGroundWeaponsLevel1', 'P2_Upgrade_ProtossGroundWeaponsLevel2', 'P2_Upgrade_ProtossGroundWeaponsLevel3', 'P2_Upgrade_ProtossShieldsLevel1', 'P2_Upgrade_ProtossShieldsLevel2', 'P2_Upgrade_ProtossShieldsLevel3', 'P2_Upgrade_PsiStormTech', 'P2_Upgrade_WarpGateResearch', 'P2_minerals', 'P2_vespene']

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


def getMmrAndResult(replayId, playerId, database):
	playerData = database.players.aggregate([
	#get matching stuff
	{ "$match" : {"replay_id" : str(replayId), "player_id" : playerId} },
	#define what to expose from the matching above.
    {"$project" : {"player_mmr" : 1, "player_id" : 1, "result" : 1}},
    ])

	for match in playerData:
		return match["player_mmr"], match["result"]


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

def extractFrameState(frameId, replayIds):
	global allColumns
	# INITIATING DATA FRAMES
	#df_p1 = pd.DataFrame(columns=allColumns) #dtype='int'
	df_p1 = pd.DataFrame(columns=["0Replay_id", "0Frame_id"]) #dtype='int'
	df_p2 = pd.DataFrame(columns=["0Replay_id", "0Frame_id"]) #dtype='int'
	#df_p2 = pd.DataFrame(columns=allColumns) #dtype='int'
	stateData = db.states.aggregate([
		  #get matching stuff
		    #{ "$match" : {"frame_id" : {"$in" : framesOfInterest }, } },
		    { "$match" : {"frame_id" : frameId} },
		  #define what to expose from the matching above.
		    {"$project" : {"resources" : 1, "replay_id": 1, "_id" : 0, "frame_id" : 1, "player_id" : 1, "supply" : 1, "units" : 1, "upgrades" : 1, "seen_enemy_units" : 1 }},
		    ])

	# PARSE STATE DATA.
	count = 0
	countRejected = 0

	for state in stateData:
	  if state["replay_id"] not in replayIds:
	  	countRejected += 1
	  else:
	  	newRow = extractData(state)
	  	#get mmr and result
	  	mmr, result = getMmrAndResult(state["replay_id"], state["player_id"], db)
	  	newRow["0P" + str(state["player_id"]) + "_mmr"] = mmr
	  	newRow["0P" + str(state["player_id"]) + "_result"] = result
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

	print("Finished " + str(frameId) +", total states parsed:", count, ", rejected: ", countRejected)

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
		
		#add all columns
		merged = merged.reindex_axis(allColumns, 'columns')

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

# turns out that about 2% of the resulting csv rows only contain data from one player. 
# such rows are not present in replay_ids
# Creating set to filter out these broken rows. 
replayIds = set({})
for replay_doc in replays.find({}, {"replay_id": 1}):
	replayIds.add(replay_doc["replay_id"])

for frame in framesOfInterest:
	print("extracting frame" + str(frame)) 
	extractFrameState(frame, replayIds)






















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