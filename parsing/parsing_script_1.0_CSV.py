import pandas as pd
import sc2reader
from pprint import pprint

replay = sc2reader.load_replay('FullGame-1.SC2Replay', load_level=4)
#replay = sc2reader.load_replay('Test-All.SC2Replay', load_level=4)
#replay = sc2reader.load_replay('Test-Archon.SC2Replay', load_level=4)
#replay = sc2reader.load_replay('Test-Building-Pylon.SC2Replay', load_level=4)
#replay = sc2reader.load_replay('Test-Upgrade-Charge.SC2Replay', load_level=4)
#replay = sc2reader.load_replay('Test-WarpGate.SC2Replay', load_level=4)
#replay = sc2reader.load_replay('Test-WarpGate2.SC2Replay', load_level=4)

# ----- INTERESTING VARIABLES -----
maxRealSeconds = 120.0
printStuff = True;
createCSV = True;


# ----- GENERAL INFORMATION OUTPRINT -----
def formatTeams(replay):
	teams = list()
	for team in replay.teams:
		players = list()
		for player in team:
			players.append("({0}) {1}".format(player.pick_race[0], player.name))
		formattedPlayers = '\n         '.join(players)
		teams.append("Team {0}:  {1}".format(team.number, formattedPlayers))
	return '\n\n'.join(teams)

def formatReplay(replay):
    return """
{filename}
--------------------------------------------
SC2 Version {release_string}
{category} Game, {start_time}
{type} on {map_name}
Length: {game_length}

{formattedTeams}
--------------------------------------------
""".format(formattedTeams=formatTeams(replay), **replay.__dict__)



# ----- PLAYER CLASS -----
class Player:
	def __init__(self, pid, name):
		self.pid = pid;
		self.name = name
		
		self.entities = {}
		self.upgrades = {}

		# UNITS
		self.entities["unitsText"] = "---UNITS:---"
		self.entities["Adept"] = 0
		self.entities["Archon"] = 0
		self.entities["Carrier"] = 0
		self.entities["Colossus"] = 0		
		self.entities["DarkTemplar"] = 0
		self.entities["Disruptor"] = 0
		self.entities["HighTemplar"] = 0
		self.entities["Immortal"] = 0		
		self.entities["Mothership"] = 0
		self.entities["Observer"] = 0
		self.entities["Oracle"] = 0
		self.entities["Phoenix"] = 0
		self.entities["Probe"] = 0
		self.entities["Sentry"] = 0
		self.entities["Stalker"] = 0
		self.entities["Tempest"] = 0
		self.entities["WarpPrism"] = 0
		self.entities["VoidRay"] = 0
		self.entities["Zealot"] = 0

		# BUILDINGS
		self.entities["buildingsText"] = "---BUILDINGS:---"
		self.entities["Assimilator"] = 0
		self.entities["CyberneticsCore"] = 0
		self.entities["DarkShrine"] = 0
		self.entities["FleetBeacon"] = 0
		self.entities["Forge"] = 0		
		self.entities["Gateway"] = 0
		self.entities["Nexus"] = 0
		self.entities["PhotonCannon"] = 0
		self.entities["Pylon"] = 0
		self.entities["RoboticsFacility"] = 0
		self.entities["RoboticsBay"] = 0
		self.entities["ShieldBattery"] = 0
		self.entities["Stargate"] = 0
		self.entities["TemplarArchive"] = 0
		self.entities["TwilightCouncil"] = 0
		self.entities["WarpGate"] = 0

		# UPGRADES
		self.entities["upgradeText"] = "---UPGRADES:---"
		self.upgrades["ProtossAirArmorsLevel1"] = False
		self.upgrades["ProtossAirArmorsLevel2"] = False
		self.upgrades["ProtossAirArmorsLevel3"] = False
		self.upgrades["ProtossAirWeaponsLevel1"] = False
		self.upgrades["ProtossAirWeaponsLevel2"] = False
		self.upgrades["ProtossAirWeaponsLevel3"] = False
		self.upgrades["ProtossGroundArmorsLevel1"] = False
		self.upgrades["ProtossGroundArmorsLevel2"] = False
		self.upgrades["ProtossGroundArmorsLevel3"] = False
		self.upgrades["ProtossGroundWeaponsLevel1"] = False
		self.upgrades["ProtossGroundWeaponsLevel2"] = False
		self.upgrades["ProtossGroundWeaponsLevel3"] = False
		self.upgrades["ProtossShieldsLevel1"] = False
		self.upgrades["ProtossShieldsLevel2"] = False
		self.upgrades["ProtossShieldsLevel3"] = False
		self.upgrades["AdeptPiercingAttack"] = False
		self.upgrades["BlinkTech"] = False
		self.upgrades["Charge"] = False
		self.upgrades["DarkTemplarBlinkUpgrade"] = False
		self.upgrades["ExtendedThermalLance"] = False
		self.upgrades["GraviticDrive"] = False
		self.upgrades["ObserverGraviticBooster"] = False
		self.upgrades["PhoenixRangeUpgrade"] = False
		self.upgrades["PsiStormTech"] = False
		self.upgrades["WarpGateResearch"] = False

# ----- CHANGEABLE TYPE PAIRS -----
typePairs = {}
typePairs["Gateway"] = "WarpGate"
typePairs["WarpGate"] = "Gateway"

# ----- TIME -----
currentBlizzardSeconds = 0.0
blizzardToRealTimeFactor = 0.725

# ----- GLOBAL VARIABLES -----
players = list()
keysNotFound = list()

# ----- ONLY FOR INITIALIZATION OF 'players' -----
teams = list()
playerNames = list()
for team in replay.teams:
	for player in team:
	    players.append(Player(player.pid, player.name))

# ---- PARSING FUNCTION -----
def parse(event, pid, name, parsingType):
	keyFound = False;
	for player in players:
		if(player.pid == pid):
			if(parsingType == "add"):
				for key in player.entities:
					if(key == name):
						player.entities[key] += 1;
						keyFound = True;
			elif(parsingType == "subtract"):
				for key in player.entities:
					if(key == name):
						player.entities[key] -= 1;
						keyFound = True;
			elif(parsingType == "enable"):
				for key in player.upgrades:
					if(key == name):
						player.upgrades[key] = True;
						keyFound = True;
	if not(keyFound):
		if(name not in keysNotFound):
			keysNotFound.append(name)
	return players




# ---- MAIN EVENT LOOP -----
for event in replay.events:

	# ----- TIME -----
	if(hasattr(event, 'second')):
		currentBlizzardSeconds = event.second
	if(maxRealSeconds > 0.0):
		if(currentBlizzardSeconds * blizzardToRealTimeFactor >= maxRealSeconds):
			break;
	
	# ----- UNITS CREATION -----
	if(isinstance(event, sc2reader.events.tracker.UnitBornEvent)):
		parse(event, event.control_pid, str(event.unit_type_name), "add")

	# ----- BUILDING AND UNIT WARP-IN -----
	elif(isinstance(event, sc2reader.events.tracker.UnitInitEvent)):
		parse(event, event.control_pid, str(event.unit_type_name), "add")

	# ----- UNIT DEATH AND BUILDING DESTRUCTION
	elif(isinstance(event, sc2reader.events.tracker.UnitDiedEvent)):
		if hasattr(event.unit.owner, "pid"):
			parse(event, event.unit.owner.pid, str(event.unit.title), "subtract")

	# ----- TYPE CHANGED EVENT ----- (Gateway <-> WarpGate)
	elif(isinstance(event, sc2reader.events.tracker.UnitTypeChangeEvent)):
		parse(event, event.unit.owner.pid, str(event.unit_type_name), "add")
		parse(event, event.unit.owner.pid, typePairs[event.unit_type_name], "subtract")

	# ----- UPGRADES -----
	elif(isinstance(event, sc2reader.events.tracker.UpgradeCompleteEvent)):
		parse(event, event.pid, str(event.upgrade_type_name), "enable")




# ----- OUTPRINTS -----
if(printStuff):
	# ----- GENERAL INFORMATION ----
	print(replay.map_name)
	print(formatReplay(replay))
	#print("--------------------------------------------")

	# ----- UNRECOGNIZED ENTITY KEYS ----
	print("Keys not parsed:"+str(keysNotFound))
	#
	print("--------------------------------------------")

	# ----- GAME STATE BY PLAYER ----
	for player in players:
		print(player.name+":")
		for key, value in player.entities.items():
			if(isinstance(value, str)):
			   print(value)
			elif(value != 0):
				print("-"+str(key)+":"+str(value))
		for key, value in player.upgrades.items():
			if(value):
				print("-"+str(key)+":"+str(value))
		print("--------------------------------------------")
		
	# ----- GAME TIME ----
	print("RealLifeSecondsPassed:"+str(currentBlizzardSeconds * blizzardToRealTimeFactor)+" | BlizzardSecondsPassed:"+str(currentBlizzardSeconds))

# ----- PRINT CSV  -----
#assumes 1v1
if(createCSV):
	#print("--------- creating .CSV -----------")
	index = [0]
	# Create dataframe containing current time, values from player 0 followed by player 1 stored in a single row. Keys as columns.
	df = pd.DataFrame([[currentBlizzardSeconds] + players[0].entities.values() + players[1].entities.values()], index=index, columns = ["BlizzardSeconds"] + players[0].entities.keys() + players[1].entities.keys())
	df = df.drop('upgradeText', 1)
	df = df.drop('buildingsText', 1)
	df = df.drop('unitsText', 1)
	print(df)
	#Save dataframe locally
	df.to_csv("FullGame-1" + ".scv")
