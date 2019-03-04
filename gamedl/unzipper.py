import os
import subprocess
from sorter import sort

mainDir = "../../replays/" #The directory containing the replay files, change if needed.
path_7z = r"C:\Program Files\7-Zip\7z.exe" #CHANGE THIS!

def unzipAll(targetPath):
    files = [f for f in os.listdir(targetPath) if os.path.isfile(os.path.join(targetPath, f)) and f.lower().endswith(".zip")]
    print("--- Unzipping", files.__len__(), "archives ---")
    successCount = 0

    for x in range(0, files.__len__()):
        file = files[x]
        print("Unzipping file", x + 1, "of", files.__len__())

        cmd = [path_7z, 'e', mainDir + str(file), '-piagreetotheeula', '-o' + mainDir, '-aos']
        sp = subprocess.check_call(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

        os.remove(os.path.join(targetPath, file))
        successCount += 1

    print("--- Done! Unzipped", files.__len__(), "archives.", successCount, "successful, probably. ---")

def main():
    unzipAll(mainDir)
    sort(mainDir)

if __name__ == "__main__":
   main()