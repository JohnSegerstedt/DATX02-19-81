import matplotlib.pyplot as plt
import numpy as np
import os
import sc2reader

mainDir = "../../replays/sorted/"

def plot_game_length(targetPath):
    files = [f for f in os.listdir(targetPath) if os.path.isfile(os.path.join(targetPath, f)) and f.lower().endswith(".sc2replay")]
    bars = []
    for x in range(0, files.__len__()):

        if x % 20 == 0:
            print("Collecting replays... :: ", round((x/files.__len__() * 100), 2), "%")

        file = files[x]
        replay = sc2reader.load_replay(targetPath + file, load_level=0)
        length = replay.game_length.seconds
        if length < 20000 and length > 5:
            found = False
            for i in range(0, bars.__len__()):
                if bars[i][0] == length:
                    found = True
                    bars[i] = (bars[i][0], bars[i][1] + 1)
            if not found:
                bars.append((length, 1))
        #else:
         #   print(length)

    bars.sort(key=lambda tup: tup[0])
    x = list(zip(*bars))
    plt.plot(x[0], x[1])
    plt.show()

def main():
    plot_game_length(mainDir)

if __name__ == "__main__":
   main()
