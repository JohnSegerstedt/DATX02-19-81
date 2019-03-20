import pandas as pd
import os

#Purges all rows where the name matches the specified purgeTarget from all data files with the specified prefix.
#Useful for removing bad or modded games :-)
#PurgeTarget needs to match the row exactly

targetPath = "../../data/"
filePrefix = "Replays6"
purgeTarget = "replaysTest/4-3-1-65094-2018-05-27T18-03-22-596.SC2Replay"

files = [f for f in os.listdir(targetPath) if os.path.isfile(os.path.join(targetPath, f)) and f.lower().endswith(".csv") and f.startswith(filePrefix)]

print("Found", files.__len__(), "files.")
for x in range(0, files.__len__()):
    file = files[x]
    data = pd.read_csv(targetPath + file)
    data = data.drop(data.columns[0], axis=1)

    amountBefore = len(data.loc[data['Name'] == purgeTarget])
    data = data[data.Name != purgeTarget]
    amountAfter = len(data.loc[data['Name'] == purgeTarget])
    if amountBefore != amountAfter:
        print("Removed", amountBefore - amountAfter, "rows from", file)

        data = data.reset_index(drop=True)
        data.to_csv(targetPath + file, index=True)
