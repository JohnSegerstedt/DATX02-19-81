import csv


# These depends on the layout of the .CSV
columnIndexMMR1 = 2
columnIndexMMR2 = 4
columnIndexReplayName = 6

# Changable variables
mmr_size = 10000
print_out = True
save_txt = True
csv_name = "test_data"
txt_name = "test_txt"


# --- MAIN ---
with open(csv_name+'.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    top_mmr = {}
    all_mmr = {}
    for row in reader:
        highest_mmr_in_replay = max(row[columnIndexMMR1], row[columnIndexMMR2])
        replayName = row[columnIndexReplayName]
        all_mmr[replayName] = highest_mmr_in_replay
    all_mmr = sorted(all_mmr.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
    for key,value in all_mmr:
        if(len(top_mmr) < mmr_size):
            top_mmr[key] = value

# --- OUTPRINT ---
if print_out:
    for replayName in top_mmr:
        print("MMR:"+top_mmr[replayName]+" | Replay:"+replayName)

# --- SAVE ---
if save_txt:
    with open(txt_name+'.txt', 'w') as f:
        for replayName in top_mmr:
            f.write(replayName+'\n')

            
