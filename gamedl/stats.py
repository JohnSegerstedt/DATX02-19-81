import matplotlib.pyplot as plt
import numpy as np
import os
import sc2reader

mainDir = "../../replays/sorted/"

#label = ['Adventure', 'Action', 'Drama', 'Comedy', 'Thriller/Suspense', 'Horror', 'Romantic Comedy', 'Musical',
        # 'Documentary', 'Black Comedy', 'Western', 'Concert/Performance', 'Multiple Genres', 'Reality']

no_movies = [
    941,
    854,
    4595,
    2125,
    942,
    509,
    548,
    149,
    1952,
    161,
    64,
    61,
    35,
    5
]

def plot_game_length(targetPath):
    files = [f for f in os.listdir(targetPath) if os.path.isfile(os.path.join(targetPath, f)) and f.lower().endswith(".sc2replay")]
    bars = []
    for x in range(0, files.__len__()):
        file = files[x]
        replay = sc2reader.load_replay(targetPath + file, load_level=0)
        length = replay.game_length.seconds
        if length < 2000 and length > 5:
            found = False
            for i in range(0, bars.__len__()):
                if bars[i][0] == length:
                    found = True
                    bars[i] = (bars[i][0], bars[i][1] + 1)
            if not found:
                bars.append((length, 1))
        else:
            print(length)

    bars.sort(key=lambda tup: tup[0])
    x = list(zip(*bars))
    plt.plot(x[0], x[1])
    plt.show()
    #index = np.arange(len(label))
    #plt.bar(index, no_movies)
    #plt.xlabel('Seconds', fontsize=5)
    #plt.ylabel('No of replays', fontsize=5)
    #plt.xticks(index, label, fontsize=5, rotation=30)
   # plt.title('Game length')
   # plt.show()

def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]

def main():
    plot_game_length(mainDir)

if __name__ == "__main__":
   main()