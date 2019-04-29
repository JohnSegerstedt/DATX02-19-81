import os
import pandas

path1 = "../reaperCSVs/cluster data 10k/"
path2 = "../reaperCSVs/vision data full/"
targetPath = "../reaperCSVs/training data full/"
targetPrefix = "training_full-"
drop_regex1 = "P2"
drop_regex2 = "P2"
drop_columns = ['Unnamed: 0']
join_column = '0Replay_id'
frame_column = '0Frame_id'

files1 = [f for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f)) and f.lower().endswith(".csv")]
if files1.__len__() == 0:
    exit(0)

files2 = [f for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f)) and f.lower().endswith(".csv")]
if files2.__len__() == 0:
    exit(0)

num_files = min(files1.__len__(), files2.__len__())
print("Found", files1.__len__(), "and", files2.__len__(), "files. Merging", num_files, "files.")

if not os.path.isdir(targetPath):
    os.makedirs(targetPath)

df1 = pandas.read_csv(os.path.join(path1, files1[0]))
df1 = df1.drop(drop_columns, axis=1)
df2 = pandas.read_csv(os.path.join(path2, files2[0]))
df2 = df2.drop(drop_columns, axis=1)

for x in range(1, num_files):
    df1x = pandas.read_csv(os.path.join(path1, files1[x]))
    df1x = df1x.drop(drop_columns, axis=1)
    df1 = df1.append(df1x, ignore_index=True)
    df2x = pandas.read_csv(os.path.join(path2, files2[x]))
    df2x = df2x.drop(drop_columns, axis=1)
    df2 = df2.append(df2x, ignore_index=True)
    print("--", x, "/", num_files, "--")

df1 = df1.drop(df1.filter(regex=drop_regex1).columns, axis=1)
df2 = df2.drop(df2.filter(regex=drop_regex2).columns, axis=1)
data = pandas.merge(df1, df2, how='inner')

print("Grouping..")
groups = data.groupby(frame_column)
print("Writing..")
count = 0
for key, df in groups:
    frame = int(df[frame_column].values[0])
    df.reset_index(drop=True, inplace=True)
    df.to_csv(targetPath + targetPrefix + str(frame) + ".csv", index=True)
    count += 1
    print("--", count, "/", num_files, "--")