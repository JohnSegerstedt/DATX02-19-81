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

def readAndPrep(dir):
    df = pd.read_csv(dir)
    del df['Unnamed: 0']
    df = df.loc[:, (df!=0).any(axis=0)]
    return df


def clusterAtTimestamp(timestamp, ids, k):
    dir = '../data/Replays6-' + str(timestamp) + 's.csv'
    df = readAndPrep(dir)
    df, nameCol = Methods.rmName(df)
    df_km = Methods.cluster_KMeans(Methods.getPCs(df, 4), k, True)
    if ids == 1:
        return df_km

    labels = df_km['Names']
    dict = Methods.makeLabelDict(nameCol, labels)
    dfs = []
    for e in ids:
        df_tmp = pd.read_csv('../data/Replays6-' + str(e) + 's.csv')
        dfs.append(Methods.labelDf(dict, df_tmp))
    return dfs


def multicluster(ids):
    dfs_list = []
    for e in ids:
        dfs_list.append(clusterAtTimestamp(e, ids, 3))
    return dfs_list

def multiplot(dfs_list):
    for e in dfs_list:
        plt.figure()
        i = 1
        for e2 in e:
            # plt.figure()
            plt.subplot(2, 3, i)
            e2 = Methods.getPCs(e2, 3)
            # Methods.project_onto_R3(e, ['PC 1', 'PC 2', 'PC 3'])
            Methods.project_onto_R2(e2, ['PC 1', 'PC 2'], False)
            i += 1
    plt.show()

def variance(df):
    clusterVars = []
    names = ['All Data']
    clusterVars.append(np.sum(np.var(df)))
    clusters = set(list(df['Names']))
    for e in clusters:
        names.append(e)
        clusterVars.append(np.sum(np.var(df[df['Names'] == e])))
    return clusterVars, names



#dfs_list = multicluster(ids = [90, 180, 270, 390, 510, 600])
#multiplot(dfs_list)
#x = clusterAtTimestamp(390, [390], 10)
df = readAndPrep('../data/Replays6-390s.csv')
cols = df.columns
for e in cols[20:]:
    del df[e]
#Methods.elbowMethod(Methods.getPCs(df,10), 100)
x = Methods.cluster_KMeans(Methods.getPCs(df, 10), 15, True)
#prevar = np.sum(np.var(x))
#x = Methods.cluster_KMeans(df, 10, True)
#x = Methods.cluster_DBSCAN(df, 1, 1, False, True)
vars, names = variance(x)
plt.plot(vars)
plt.show()
print('hej')













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
