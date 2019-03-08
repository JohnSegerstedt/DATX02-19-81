import os.path
import pandas as pd
import sc2reader
from pprint import pprint

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

def mainFunction(line, startTime, interval, endTime, destination):
    # ----- INPUT VARIABLES -----
        replayName = line
        nextInterval = startTime #NOTE: If interval = 0, only one timestamp will be printed and that is startTime.
        destination = destination

        #print("Replay: "+replayName)
        #print("Time: "+str(maxRealSeconds))
        #print("Destination: "+destination)

        replay = sc2reader.load_replay(replayName, load_level=4)

        # ----- INTERESTING VARIABLES -----
        printStuff = False;
        createCSV = True;

        # ----- PLAYER CLASS -----
        class Player:
                def __init__(self, pid, name):
                        self.pid = pid;
                        self.name = name
                        
                        self.entities = {}
                        self.upgrades = {}

                        # UNITS
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
                        self.entities["ProtossAirArmorsLevel1"] = 0
                        self.entities["ProtossAirArmorsLevel2"] = 0
                        self.entities["ProtossAirArmorsLevel3"] = 0
                        self.entities["ProtossAirWeaponsLevel1"] = 0
                        self.entities["ProtossAirWeaponsLevel2"] = 0
                        self.entities["ProtossAirWeaponsLevel3"] = 0
                        self.entities["ProtossGroundArmorsLevel1"] = 0
                        self.entities["ProtossGroundArmorsLevel2"] = 0
                        self.entities["ProtossGroundArmorsLevel3"] = 0
                        self.entities["ProtossGroundWeaponsLevel1"] = 0
                        self.entities["ProtossGroundWeaponsLevel2"] = 0
                        self.entities["ProtossGroundWeaponsLevel3"] = 0
                        self.entities["ProtossShieldsLevel1"] = 0
                        self.entities["ProtossShieldsLevel2"] = 0
                        self.entities["ProtossShieldsLevel3"] = 0
                        self.entities["AdeptPiercingAttack"] = 0
                        self.entities["BlinkTech"] = 0
                        self.entities["Charge"] = 0
                        self.entities["DarkTemplarBlinkUpgrade"] = 0
                        self.entities["ExtendedThermalLance"] = 0
                        self.entities["GraviticDrive"] = 0
                        self.entities["ObserverGraviticBooster"] = 0
                        self.entities["PhoenixRangeUpgrade"] = 0
                        self.entities["PsiStormTech"] = 0
                        self.entities["WarpGateResearch"] = 0

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
                if not(keyFound):
                        if(name not in keysNotFound):
                                keysNotFound.append(name)
                return players

        def printInfo():
                # ----- GENERAL INFORMATION ----
                print(replay.map_name)
                print(formatReplay(replay))
                # print("--------------------------------------------")

                # ----- UNRECOGNIZED ENTITY KEYS ----
                print("Keys not parsed:" + str(keysNotFound))
                #
                print("--------------------------------------------")

                # ----- GAME STATE BY PLAYER ----
                for player in players:
                        print(player.name + ":")
                        for key, value in player.entities.items():
                                if (isinstance(value, str)):
                                        print(value)
                                elif (value != 0):
                                        print("-" + str(key) + ":" + str(value))

                # ----- GAME TIME ----
                print("RealLifeSecondsPassed:" + str(
                        currentBlizzardSeconds * blizzardToRealTimeFactor) + " | BlizzardSecondsPassed:" + str(
                        currentBlizzardSeconds))

        def printCSV(timeStamp):
                dataIsGood = True;
                for player in players:
                        if (player.entities["Probe"] == 0):
                                dataIsGood = False;
                if (dataIsGood):
                        # creating a list for p1 to ensure unique column names.
                        p1keylist = [x + "1" for x in list(players[0].entities.keys())]
                        if (printStuff):
                                print(p1keylist)
                        index = [0]
                        replayNameList = [replayName]
                        nameTitleList = ["Name"]
                        # Create dataframe containing current time, values from player 0 followed by player 1 stored in a single row. Keys as columns.
                        df = pd.DataFrame(
                                [replayNameList + list(players[0].entities.values()) + list(players[1].entities.values())],
                                index=index, columns=nameTitleList + list(players[0].entities.keys()) + p1keylist)
                        fileName = str(destination) + "-" + str(timeStamp) + "s.csv"
                        # Read from file
                        if os.path.isfile(fileName):
                                try:
                                        temp = pd.read_csv(fileName)
                                        df = pd.concat([df, temp], join_axes=[df.columns], ignore_index=True)
                                except pd.io.common.EmptyDataError:
                                        print("file read error")

                        df.to_csv(fileName)
                        if (printStuff):
                                print(df)



        # ---- MAIN EVENT LOOP -----
        for event in replay.events:

                #print(event.name)

                # ----- TIME -----
                if(hasattr(event, 'second')):
                        currentBlizzardSeconds = event.second
                if(endTime > 0.0):
                        if(currentBlizzardSeconds * blizzardToRealTimeFactor >= nextInterval):
                                if createCSV:
                                        printCSV(nextInterval)
                                nextInterval += interval
                                if(nextInterval > endTime or interval == 0):
                                        if printStuff:
                                                printInfo()
                                        break
                
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
                        if hasattr(event.unit.owner, "pid"):
                                if(event.unit_type_name in typePairs):
                                        parse(event, event.unit.owner.pid, str(event.unit_type_name), "add")
                                        parse(event, event.unit.owner.pid, typePairs[event.unit_type_name], "subtract")
                                else:
                                        if(printStuff):
                                                print("Does not exist in typePairs: '"+str(event.unit_type_name)+"'")

                # ----- UPGRADES -----
                elif(isinstance(event, sc2reader.events.tracker.UpgradeCompleteEvent)):
                        parse(event, event.pid, str(event.upgrade_type_name), "add")



#mainFunction("b98ea23acc0eccd264853fce10ede38724aba594e729b62fbf76035d0cdc532b.SC2Replay", 60, "test")
#mainFunction("FullGame-1.SC2Replay", 60, "test")

