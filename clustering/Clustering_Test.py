from clustering.Utilities import Methods
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from clustering.Utilities.som import SOM
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning) #Supresses those pesky warnings

# For plotting the images
from matplotlib import pyplot as plt


def labelDf(labelDict, df_unlabeled):

    newNames = []
    for e in df_unlabeled['Name']:
        if labelDict.__contains__(e):
            newNames.append(labelDict.get(e))
        else:
            newNames.append('Null')
    df_unlabeled['Names'] = newNames
    del df_unlabeled['Unnamed: 0']
    del df_unlabeled['Name']
    df_tmp = df_unlabeled.loc[:, (df_unlabeled != 0).any(axis=0)]
    return df_tmp[df_tmp['Names'] != 'Null']

def getReplays(df, names):
    return df.loc[df['Name'].isin(names)]

def getColumn(df, column):
    return df.loc[df['column']]


def makeLabelDict(names, labels):
    dict = {}
    for i in range(0, len(labels)):
        dict[names[i]] = labels[i]
    return dict

def clusterAtTimestamp(timestamp, ids):
    df = pd.read_csv('../data/Replays6-' + str(timestamp) + 's.csv')
    names = df['Name']
    del df['Unnamed: 0']
    del df['Name']
    df = df.loc[:, (df != 0).any(axis=0)]
    df_km = Methods.cluster_KMeans(Methods.getPCs(df, 4), 3, True)

    labels = df_km['Names']
    dict = makeLabelDict(names, labels)

    dfs = []
    for e in ids:
        df_tmp = pd.read_csv('../data/Replays6-' + str(e) + 's.csv')
        dfs.append(labelDf(dict, df_tmp))
    return dfs

def multicluster(ids):
    i = 1
    dfs_list = [clusterAtTimestamp(90, ids), clusterAtTimestamp(180, ids), clusterAtTimestamp(270, ids), clusterAtTimestamp(390, ids), clusterAtTimestamp(510, ids), clusterAtTimestamp(600, ids)]
    for e in dfs_list:
        plt.figure()
        i = 1
        for e2 in e:
            #plt.figure()
            plt.subplot(2, 3, i)
            e2 = Methods.getPCs(e2, 3)
            #Methods.project_onto_R3(e, ['PC 1', 'PC 2', 'PC 3'])
            Methods.project_onto_R2(e2, ['PC 1', 'PC 2'], False)
            i += 1
    plt.show()
    return dfs_list

dfs_list = multicluster(ids = [90, 180, 270, 390, 510, 600])
print()
''' #Slafsig plot
del df['Unnamed: 0']
del df['Name']
df = df.loc[:, (df != 0).any(axis=0)]
for e in df.columns:
    plt.plot(df[e])
plt.show()
'''


'''
 #Intressant? Olika spelarnas tidsutveckling
id = [90, 180, 270, 390, 510, 600]
i = 1
keepCols = ['Probe','Adept','Archon','Carrier','Colossus','DarkTemplar','Disruptor','HighTemplar','Immortal','Mothership','Observer','Oracle','Phoenix']
#keepCols = ['Probe', 'Adept', ]
for e in id:
    plt.subplot(2, 3, i)
    plt.title('Timestamp: ' + str(id[i-1]) + 's.')
    i += 1
    loaddir = '../data/Replays6-' + str(e) + 's.csv'
    df_tmp = pd.read_csv(loaddir)
    del df_tmp['Unnamed: 0']
    del df_tmp['Name']
    #df_tmp = df_tmp.loc[:, (df_tmp != 0).any(axis=0)]
    for e in df_tmp.columns:
        if e not in keepCols:
            del df_tmp[e]

    df_tmp = Methods.getPCs(df_tmp, 10)
    df_tmp = Methods.cluster_DBSCAN(df_tmp, 2, 20, False, True)
    #Methods.radViz(df_tmp)
    df_km = Methods.cluster_KMeans(df_tmp, 7, True)
    Methods.project_onto_R2(df_km, ['PC 1', 'PC 2'], False)
    #Methods.parallelCoordinates_Clusters(df_km)

plt.figure()
i = 1
keepCols = ['Probe1','Adept1','Archon1','Carrier1','Colossus1','DarkTemplar1','Disruptor1','HighTemplar1','Immortal1','Mothership1','Observer1','Oracle1','Phoenix1']
#keepCols = ['Probe', 'Adept', ]
for e in id:
    plt.subplot(2, 3, i)
    plt.title('Timestamp: ' + str(id[i-1]) + 's.')
    i += 1
    loaddir = '../data/Replays6-' + str(e) + 's.csv'
    df_tmp = pd.read_csv(loaddir)
    del df_tmp['Unnamed: 0']
    del df_tmp['Name']
    #df_tmp = df_tmp.loc[:, (df_tmp != 0).any(axis=0)]
    for e in df_tmp.columns:
        if e not in keepCols:
            del df_tmp[e]

    df_tmp = Methods.getPCs(df_tmp, 10)
    df_tmp = Methods.cluster_DBSCAN(df_tmp, 2, 20, False, True)
    #Methods.radViz(df_tmp)
    df_km = Methods.cluster_KMeans(df_tmp, 7, True)
    Methods.project_onto_R2(df_km, ['PC 1', 'PC 2'], False)
    #Methods.parallelCoordinates_Clusters(df_km)

plt.show()
'''
