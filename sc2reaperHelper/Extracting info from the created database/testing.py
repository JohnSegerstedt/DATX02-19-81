import pymongo
import enum
import pprint
import pandas as pd
import datetime

#Generates .CSV from connected MongoDB Server.

# INTERESTING VARIABLES
# frames come in sizes 12*x
framesOfInterest = [12000] #1200,2400,3600]
databaseName = "replay_database_test"

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

class rowIndex(enum.IntEnum):
  Adept = 0
  AdeptPhaseShift = 1
  Archon = 2
  Assimilator = 3
  Carrier = 4
  Colossus = 5
  CyberneticsCore = 6
  DarkShrine = 7
  DarkTemplar = 8
  Disruptor = 9
  DisruptorPhased = 10
  FleetBeacon = 11
  ForceField = 12
  Forge = 13
  Gateway = 14
  HighTemplar = 15
  Immortal = 16
  Interceptor = 17
  Mothership = 18
  MothershipCore = 19
  Nexus = 20
  Observer = 21
  ObserverSurveillanceMode = 22
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
  global df_test
  global df_p1
  global df_p2
  global df
  df = df.append(data, ignore_index=True)
  df_test = df_test.append(data, ignore_index=True)

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
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#    print(df_p1)
df_test = pd.DataFrame()

# INITIATING SORTING TUPLES
dictUpgradeIndex = {}
i = 0
for e in UpgradeId:
  dictUpgradeIndex[e.value] = i 
  i+=1

dictUnitIndex = {}
i = 0
for e in UnitId:
  dictUnitIndex[e.value] = i 
  i+=1

# TODO generalize for all games, and for each state of interest
  # TODO iterate through each frame, count spotted units of each type, keeping track of max so far, for one game up until state of interest
  # states_visible_enemy_units = []

states = db.states.aggregate([
  #get matching stuff
    { "$match" : {"frame_id" : {"$in" : framesOfInterest }, } },
  #define what to expose from above match.
    {"$project" : {"replay_id": 1, "_id" : 0, "frame_id" : 1, "player_id" : 1, "units" : 1, "upgrades" : 1 }},
    ])

count = 0
for state in states:
  newRow = extractData(state)
  saveToDataframe(newRow, state["player_id"])
  count += 1
  #print("States saved:", count)

print("Finished, total states parsed:", count)

print("FINAL OUTPUT test")
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#    print(df_test)

printCSV(df)

#pprint.pprint(df)
#TODO remove NaNs
print(list(df_test.columns.values))
