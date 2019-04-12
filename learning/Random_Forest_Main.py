
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import os
import random
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning) #Supresses those pesky warnings
from matplotlib import pyplot as plt



def filter_by_frame(files, max_frame):
    return [f for f in files if int(f[f.index("-") + 1:-4]) <= max_frame]



def load_data_over_time(path, targetsFile): #Should return: data array, target array
    if not os.path.isdir(path):
        exit(0)
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(".csv")]
    if files.__len__() == 0:
        exit(0)

    files = filter_by_frame(files, frame_cutoff)
    print("Found", files.__len__(), "files.")
    data = pd.read_csv(os.path.join(path, files[0]))
    data = data.drop(drop_columns, axis=1)
    for x in range(1, files.__len__()):
        df = pd.read_csv(os.path.join(path, files[x]))
        df = df.drop(drop_columns, axis=1)
        df = df.drop(drop_columns_2, axis=1)
        data = pd.merge(data, df, on=join_column, how='inner', suffixes=['_' + str(x - 1), '_' + str(x)])

    data = pd.merge(data, pd.read_csv(targetsFile), on=join_column, how='inner')
    global distribution
    global num_classes
    num_classes = int(data[target_column].max() + 1)
    data = data.drop(join_column, axis=1)
    #print(list(data.columns.values))
    #exit(0)
    targets = data.filter([target_column], axis=1)
    data = data.drop(target_column, axis=1)

    #data = data.drop(data.filter(regex='P1').columns, axis=1)

    data = data.values
    data = MinMaxScaler().fit_transform(data)

    targets[target_column] = targets[target_column]
    targets = targets.values
    #targets = keras.utils.to_categorical(targets, num_classes)

    global input_shape
    input_shape = [len(data[0])]
    return data, targets



def mislabel(labels, p):
    labelSet = list(set(labels))
    labels = list(labels)
    for i in range(0, len(labels)):
        if random.uniform(0, 1) < p:
            tmpSet = []
            for j in range(0,len(labelSet)):
                tmpSet.append(labelSet[j])
            #tmpSet.remove(labels[i])
            labels[i] = tmpSet[random.randint(0, len(tmpSet)-1)]
    return labels

file = "../reaperCSVs/cluster data 10k/" #File to parse and use for training
#file = "../reaperCSVs/cluster data/cluster_data10080.csv"
targetsFile = "10kclustering2.csv" #File containing results from clustering, to use as targets
join_column = '0Replay_id' #Column to use as identifier when joining the files. Joining is done before dropping
drop_columns = ['Unnamed: 0', '0P1_mmr', '0P2_mmr', '0P1_result', '0P2_result', ] #Drop these columns from the original csv-file because they're irrelevant
drop_columns_2 = ['0Frame_id']
target_column = 'Cluster' #Name of the column containing the training targets (labels)
frame_cutoff = 6480

a, b = load_data_over_time(file, targetsFile)

# Targets are made and put in DF in accord with clustering
targets = np.zeros(len(b))
for i in range(0, len(b)):
    targets[i] = b[i][0]

df = pd.DataFrame(a)
df = df.loc[:, (df != 0).any(axis=0)]
df['labels'] = pd.factorize(targets)[0]


max_iter = 10
pmax = 1
pmin = 0
ass = np.linspace(pmin,pmax, max_iter)
succ = np.zeros(max_iter)
iter = 0

print('Running ' + str(max_iter) + ' iterations for p(mislabeling) with max p = ' + str(pmax)
      + ', and ' + str(len(df)) + ' datapoints.')
print()
print()

for dick in ass:

    #Partitions observations to training data and validation data subsets
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .2  # ~2% validerigsdata
    train, test = df[df['is_train'] == True], df[df['is_train'] == False]


    #Initializes classifier and trains on training data
    features = df.columns[:252]
    clf = None
    clf = RandomForestClassifier(n_jobs = 2, random_state = 0, n_estimators = 100)
    clf.fit(train[features], mislabel(train['labels'], dick))  # trains the classifier
    #clf = KNeighborsClassifier(n_neighbors=50)  # Initializes KNN classifier, n_jobs: parralelizes
    #clf.fit(train[features], mislabel(train['labels'], dick))



    # predicts with the trained model
    y_pred = clf.predict(test[features])
    accuracy = clf.score(test[features], test['labels'], sample_weight=None)
    succ[iter] = accuracy

plt.plot(ass, succ)
plt.show()

