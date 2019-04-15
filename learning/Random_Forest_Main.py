
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



def load_data_over_time(path, targetsFile, frame_cutoff): #Should return: data array, target array
    if not os.path.isdir(path):
        exit(0)
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(".csv")]
    if files.__len__() == 0:
        exit(0)

    files = filter_by_frame(files, frame_cutoff)
    print("Found", files.__len__(), "files.")
    data = pd.read_csv(os.path.join(path, files[0]))
    data = data.drop(drop_columns, axis=1)
    data = data.drop(drop_columns_2, axis=1)
    for x in range(1, files.__len__()):
        df = pd.read_csv(os.path.join(path, files[x]))
        df = df.drop(drop_columns, axis=1)
        df = df.drop(drop_columns_2, axis=1)
        data = pd.merge(data, df, on=join_column, how='inner', suffixes=['_' + str(x - 1), '_' + str(x)])
    colnames = data.columns[1:]
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
    return data, targets, colnames



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



def mislabeling(max_iter, p_min, p_max):

    a, b, colnames = load_data_over_time(file, targetsFile, frame_cutoff = 6480)

    # Targets are made and put in DF in accord with clustering
    targets = np.zeros(len(b))
    for i in range(0, len(b)):
        targets[i] = b[i][0]

    df = pd.DataFrame(a)
    df = df.loc[:, (df != 0).any(axis=0)]
    df['labels'] = pd.factorize(targets)[0]

    P = np.linspace(p_min,p_max, max_iter)
    A = np.zeros(max_iter)
    iter = 0


    print('Running ' + str(max_iter) + ' iterations for p(mislabeling) with max p = ' + str(p_max)
          + ', and ' + str(len(df)) + ' datapoints.')
    print()
    print()

    for p in P:

        #Partitions observations to training data and validation data subsets
        df['is_train'] = np.random.uniform(0, 1, len(df)) <= .8  # ~20% validerigsdata
        train, test = df[df['is_train'] == True], df[df['is_train'] == False]


        #Initializes classifier and trains on training data
        features = df.columns[:251]
        clf = None
        clf = RandomForestClassifier(n_jobs = 2, random_state = 0, n_estimators = 100)
        clf.fit(train[features], mislabel(train['labels'], p))  # trains the classifier
        #clf = KNeighborsClassifier(n_neighbors=50)  # Initializes KNN classifier, n_jobs: parralelizes
        #clf.fit(train[features], mislabel(train['labels'], dick))


        # predicts with the trained model
        y_pred = clf.predict(test[features])
        accuracy = clf.score(test[features], test['labels'], sample_weight=None)
        A[iter] = accuracy
        iter += 1

        print('Done with iter' + str(iter) + '/' + str(max_iter) + ' (p(misl) = ' + str(int(p*1000)/1000) + '). Model accuracy  was ' + str(accuracy) + '.')
        print()

    plt.plot(P, A)
    plt.show()


def frame_cutoff(F, trees):

    A = np.zeros(len(F))
    iter = 0
    C = []
    C_imp = []


    for f in F:

        a, b, colnames = load_data_over_time(file, targetsFile, f)
        C.append(colnames)

        # Targets are made and put in DF in accord with clustering
        targets = np.zeros(len(b))
        for i in range(0, len(b)):
            targets[i] = b[i][0]
        df = pd.DataFrame(a)
        #df = df.loc[:, (df != 0).any(axis=0)]
        features = df.columns[:]
        df['labels'] = pd.factorize(targets)[0]

        #Partitions observations to training data and validation data subsets
        df['is_train'] = np.random.uniform(0, 1, len(df)) <= .8  # ~20% validerigsdata
        train, test = df[df['is_train'] == True], df[df['is_train'] == False]


        #Initializes classifier and trains on training data
        clf = None
        clf = RandomForestClassifier(n_jobs = 2, n_estimators = trees)
        clf.fit(train[features], train['labels'])  # trains the classifier

        C_imp.append(clf.feature_importances_)

        accuracy = clf.score(test[features], test['labels'], sample_weight=None)
        A[iter] = accuracy
        iter += 1
        print('Done with iter ' + str(iter) + ' (Frames until and with' + str(f) + '). Model accuracy was ' + str(accuracy) + '.')
        print()

    return F, A, C, C_imp



file = "../reaperCSVs/cluster data 10k/" #File to parse and use for training
#file = "../reaperCSVs/cluster data/cluster_data10080.csv"
targetsFile = "10kclustering2.csv" #File containing results from clustering, to use as targets
join_column = '0Replay_id' #Column to use as identifier when joining the files. Joining is done before dropping
drop_columns = ['Unnamed: 0', '0P1_mmr', '0P2_mmr', '0P1_result', '0P2_result'] #Drop these columns from the original csv-file because they're irrelevant
drop_columns_2 = ['0Frame_id']
target_column = 'Cluster' #Name of the column containing the training targets (labels)

Atot = []
t = np.arange(1,300,3)
for i in t:
    print(i)
    F = np.arange(5760, 5760+720, 720)
    F, A, C, C_imp = frame_cutoff(F, i)
    Atot.append(A[0])

print()
'''
params = C[0][0:146]
imps = np.zeros(146)
for i in range(0, len(C_imp[0])):
    imps[i%146] += C_imp[0][i]

imps_big = []
params_big = []
for i in range(0, len(imps)):
    if imps[i] > 0.015:
        imps_big.append(imps[i])
        params_big.append(params[i])
imps_big = imps_big/np.max(imps_big)
plt.barh(params_big, imps_big)

plt.title('Relative feature importance in RF for ~20 most imp. (T = 5760, accuracy ~ 97%)')
plt.xlabel('Relative feature importance')
plt.ylabel('Feature')

plt.show()

print()
#mislabeling(10, 0, 1)
'''
