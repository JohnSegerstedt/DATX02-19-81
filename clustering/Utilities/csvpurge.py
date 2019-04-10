import pandas as pd
import os

#Purges all rows where the name matches the specified purgeTarget from all data files with the specified prefix.
#Useful for removing bad or modded games :-)
#PurgeTarget needs to match the row exactly

targetPath = "../../reaperCSVs/cluster data 40k/"
filePrefix = ""

purgeRows = False
purgeTarget = ""
purgeColumn = 'Name'

purgeColumns = False
columnsToPurge = ['Unnamed: 0.1', 'Unnamed: 0']

purgeDuplicates = True
duplicatesColumn = '0Replay_id'

files = [f for f in os.listdir(targetPath) if os.path.isfile(os.path.join(targetPath, f)) and f.lower().endswith(".csv") and f.startswith(filePrefix)]

print("Found", files.__len__(), "files.")
for x in range(0, files.__len__()):
    file = files[x]
    data = pd.read_csv(targetPath + file)
    data = data.drop('Unnamed: 0', axis=1)

    shapeBefore = data.shape

    if purgeColumns:
        cols = [c for c in columnsToPurge if c in data.columns]
        data = data.drop(cols, axis=1)

    if purgeDuplicates:
        data = data.drop_duplicates([duplicatesColumn], keep='first')
    if purgeRows:
        data = data[data.Name != purgeTarget]
    shapeAfter = data.shape
    if shapeBefore != shapeAfter:
        print("Removed", shapeBefore[0] - shapeAfter[0], "rows and", shapeBefore[1] - shapeAfter[1], "columns from", file)

        data = data.reset_index(drop=True)
        data.to_csv(targetPath + file, index=True)
