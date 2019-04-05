# CLEANING THE MONGODB
import pymongo

# Every 720 frame_id's contains unit data
framesOfInterest = list(range(0, 21600, 720))
databaseName = "reaping2"

def connectToDatabase(db):
  # Create mongodb client
  client = pymongo.MongoClient()
  # Connect to db using client
  db = client[db]
  return db

def removeBrokenFileData(replayId):
    query = {"replay_id": replayId}
    # allCollections = ["actions", "players", "replays", "scores", "states"]

    for collection in db.list_collection_names():
        # delete in collection:
        result = db[collection].delete_many(query)
        print(result.deleted_count, " documents deleted in ", collection)

def hasZeroUnits(state):
	unitCount = 0
	for unitKey in state["units"].keys():
		unitCount += len(state["units"][unitKey])
		if unitCount >= 0:
			return False
	return unitCount == 0

def checkValidity1(frameId, replayIds):
	stateData = db.states.aggregate([
		  #get matching stuff
		    #{ "$match" : {"frame_id" : {"$in" : framesOfInterest }, } },
		    { "$match" : {"frame_id" : frameId} },
		  #define what to expose from the matching above.
		    {"$project" : {"replay_id": 1}},
		    ])

	count = 0
	countRejected = 0
	for state in stateData:
		#if replayId not in replayIds then it is broken
		#TODO fix so that replayIds is updated before continuing
	  replayId = state["replay_id"]
	  if replayId not in replayIds:
	  	countRejected += 1
	  	removeBrokenFileData(replayId)
	  	#replayIds.remove(replayId)
	  	print("removed " , replayId, " from replayIds")
	  count += 1
	  if (count%20==0):
	  	print(count)

	print("Finished frame " + str(frameId) +", total states parsed:", count, ", rejected in during this frame: ", countRejected)

def checkValidity2(frameId, replayId):
	global count
	global countRejected
	count += 1
	results = db.states.find({"frame_id" : frameId, "replay_id" : replayId}, {"replay_id": 1})#.sort("abc",pymongo.DESCENDING).skip((page-1)*num).limit(num)
	results_count = results.count()
	if results_count == 1:
		print("data from only one of the players detected, removing files")
		removeBrokenFileData(replayId)
		print("removed " , replayId, " from replayIds")
		countRejected += 1

def checkValidity3(frameId):
	# TODO gå igenom varje frame of interest och kolla för varje spelare om summan av all state data <=0, isf ta bort 
	stateData = db.states.aggregate([
		  #get matching stuff
		    #{ "$match" : {"frame_id" : {"$in" : framesOfInterest }, } },
		    { "$match" : {"frame_id" : frameId} },
		  #define what to expose from the matching above.
		    {"$project" : {"replay_id": 1, "units" : 1 }},
		    ])

	countRejected = 0
	count = 0
	for state in stateData:
		if hasZeroUnits(state):
			removeBrokenFileData(state["replay_id"])
			print("removed " , replayId, " from replayIds")
			countRejected += 1
		count += 1
		if (count%100==0):
			print(count)

	print("Finished validity check 3 for frame_id " + str(frameId) +", total states parsed:", str(count), ", rejected: ", str(countRejected))


# ---- MAIN ----
db = connectToDatabase(databaseName)
# keep track of removed entries
countRejected = 0

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


# turns out that about 2% of the resulting csv rows only contain data from one player. 
# such rows are not present in replay_ids
# Creating set to filter out these broken rows. 
replayIds = set({})
for replay_doc in replays.find({}, {"replay_id": 1}):
	replayIds.add(replay_doc["replay_id"])

print("Cleanup stage 1/3 commencing...")
for frame in framesOfInterest:
	print('checking that all replay_ids that exist in state["states"] also exist in state["replays"] for frame' + str(frame)) 
	checkValidity1(frame, replayIds)

# remove all entries where we have data from only one out of the two players in a frame_id, replay_id
print("Cleanup stage 2/3 commencing...")
replayIdsInStates = set({})
for replay_doc in states.find({}, {"replay_id": 1}):
	replayIdsInStates.add(replay_doc["replay_id"])

count = 0
countRejected = 0 
for frame in framesOfInterest:
	print("checking that frame" + str(frame) +  " contains data from both players")
	for replayId in replayIdsInStates:
		#print("check that replay id: " + str(replayId) + " contains player data from both players for frame: " + str(frame))
		checkValidity2(frame, replayId)
	print("Cleanup stage frame " + str(frame) + "completed")
print("Cleanup stage 2/3 finised, count: " + str(count) + ", rejected: " + str(countRejected))

# Ensure that no data entries contain zero valued units for all units
print("Cleanup stage 3/3 commencing...")
for frame in framesOfInterest:
	print("Stage 3: checking unit and upgrade validity for frame" + str(frame)) 
	checkValidity3(frame)