import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA, SparsePCA
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pointbiserialr
from pandas.plotting import parallel_coordinates
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.spatial.distance as ssd
from scipy.spatial import distance_matrix
from sklearn import manifold

clustering =[1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 2, 0, 0,
             0, 0, 2, 2, 1, 0, 0, 0, 1, 2, 1, 1, 1, 1, 2, 1, 0, 0, 0, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1,
             2, 1, 1, 1, 2, 0, 2, 2, 1, 2, 1, 1, 1, 1, 0, 0, 1, 2, 2, 1, 2, 2, 0, 1, 2, 0, 0, 2, 2, 1, 2, 1,
             1, 1, 0, 0, 0, 2, 2, 0, 1, 0, 0, 2, 2, 1, 0, 2, 1, 2, 2, 1, 2, 2, 0, 2, 1, 1, 1, 2, 2, 2, 2, 0,
             0, 1, 2, 1, 2, 2, 2, 0, 2, 0, 2, 2, 1, 0, 2, 0, 0, 1, 2, 1, 2, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 2,
             0, 2, 1, 0, 1, 0, 1, 2, 1, 2, 1, 0, 2, 2, 2, 0, 2, 0, 1, 1, 2, 0, 1, 2, 1, 1, 2, 2, 1, 0, 2, 1,
             2, 1, 0, 2, 0, 1, 2, 0, 2, 1, 1, 1, 1, 2, 0, 0, 0, 1, 1, 2, 0, 1, 2, 1, 0, 2, 0, 0, 2, 2, 1, 2,
             0, 2, 2, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0, 0, 1, 1, 0, 2, 0, 2, 1, 0, 1, 1, 1, 2, 0, 1, 2,
             1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 2, 0, 0, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 0, 0, 1, 0, 2, 1, 1,
             2, 1, 1, 1, 2, 2, 1, 2, 1, 0, 2, 0, 0, 1, 0, 2, 2, 1, 0, 2, 0, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2,
             1, 2, 0, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 0, 1, 2, 2, 2, 2, 1, 0, 2, 2, 1, 0,
             1, 1, 0, 1, 2, 0, 1, 0, 0, 1, 0, 2, 1, 1, 1, 2, 1, 0, 0, 2, 1, 2, 2, 2, 2, 0, 2, 1, 1, 2, 0, 1,
             0, 1, 0, 1, 1, 0, 1, 1, 0, 2, 2, 1, 0, 0, 1, 1, 0, 1, 0, 2, 1, 2, 1, 1, 2, 2, 2, 2, 0, 0, 0, 2,
             1, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 1, 1, 2, 1, 0, 1, 1, 2, 2, 1, 2, 0, 1, 0, 2, 1,
             2, 1, 0, 0, 1, 1, 0, 0, 2, 2, 0, 2, 1, 0, 2, 1, 2, 1, 2, 0, 2, 0, 2, 2, 2, 1, 0, 1, 2, 1, 0, 0,
             2, 1, 0, 0, 0, 2, 0, 1, 1, 1, 1, 2, 0, 0, 2, 0, 2, 2, 0, 0, 0, 1, 1, 2, 0, 0, 2, 2, 1, 0, 0, 2,
             0, 1, 0, 2, 1, 2, 0, 2, 1, 0, 2, 2, 2, 2, 0, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 0, 1, 0, 2, 2, 0, 2,
             2, 1, 2, 1, 2, 1, 1, 2, 0, 1, 2, 2, 1, 1, 0, 2, 0, 1, 1, 0, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 0, 1,
             2, 1, 2, 0, 1, 2, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 2, 1, 0, 1, 2, 0, 0, 0, 0, 2, 0, 1, 1, 1, 1,
             1, 0, 2, 1, 1, 1, 2, 1, 0, 1, 0, 2, 0, 1, 2, 1, 0, 1, 0, 1, 0, 0, 0, 2, 1, 1, 0, 2, 2, 1, 2, 1,
             1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 0, 1, 0, 0, 1, 1, 0, 2, 0, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 0, 2,
             0, 0, 1, 0, 2, 1, 1, 0, 0, 1, 0, 1, 0, 1, 2, 0, 2, 0, 1, 2, 0, 1, 0, 2, 1, 1, 2, 1, 2, 0, 2, 0,
             1, 1, 1, 0, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 1, 0, 2, 0, 1, 1, 2, 1, 0, 0, 0, 2, 1, 2, 0, 1,
             0, 1, 0, 2, 1, 0, 1, 2, 1, 2, 0, 0, 0, 0, 1, 0, 1, 2, 2, 0, 1, 0, 2, 2, 0, 2, 2, 1, 1, 1, 2, 2,
             1, 1, 1, 1, 2, 2, 0, 1, 2, 0, 2, 2, 0, 1, 1, 1, 0, 2, 2, 1, 0, 2, 0, 0, 0, 2, 1, 0, 2, 2, 2, 1,
             0, 1, 0, 0, 2, 0, 2, 1, 0, 0, 0, 0, 2, 2, 1, 0, 1, 0, 1, 2, 1, 1, 0, 2, 0, 0, 0, 2, 2, 2, 1, 0,
             0, 2, 2, 0, 2, 0, 2, 2, 1, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2, 1, 0, 2, 1, 0, 1, 1, 2, 0, 0, 0, 0,
             1, 2, 1, 0, 0, 1, 2, 0, 2, 1, 2, 2, 0, 2, 0, 1, 0, 0, 2, 1, 1, 0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2,
             1, 0, 1, 2, 2, 0, 2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 2, 2, 0, 2, 1, 0, 1, 2, 0, 2, 2,
             1, 1, 0, 2, 2, 1, 0, 1, 2, 1, 0, 2, 1, 2, 1, 0, 1, 2, 2, 2, 2, 2, 2, 0, 2, 2, 1, 1, 0, 1, 2, 1,
             1, 2, 0, 2, 0, 0, 2, 2, 0, 0, 1, 0, 1, 1, 0, 2, 0, 0, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 0, 2, 1,
             0, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 2, 1, 1, 2, 2, 1, 1, 1, 0, 2, 2, 2, 1, 1, 2, 0, 1, 1, 1, 1, 2,
             2, 0, 1, 2, 2, 0, 2, 1, 2, 2, 2, 0, 0, 1, 2, 1, 0, 1, 2, 2, 2, 1, 1, 2, 0, 1, 2, 0, 2, 1, 0, 2,
             0, 1, 0, 0, 0, 1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 2, 2, 1, 1, 0, 1, 0, 1, 2, 2, 2, 0, 2,
             2, 0, 0, 2, 2, 0, 2, 1, 0, 1, 2, 1, 1, 2, 0, 1, 2, 2, 1, 2, 0, 1, 0, 0, 0, 2, 2, 2, 2, 1, 1, 1,
             2, 1, 1, 1, 2, 2, 1, 1, 0, 2, 1, 2, 2, 1, 1, 0, 0, 1, 2, 1, 2, 1, 2, 2, 0, 1, 0, 1, 2, 2, 2, 1,
             2, 0, 1, 0, 2, 2, 2, 2, 2, 0, 1, 0, 1, 1, 2, 0, 2, 0, 2, 1, 0, 0, 2, 1, 0, 2, 0, 1, 1, 2, 1, 2,
             1, 1, 1, 1, 1, 2, 0, 2, 2, 2, 1, 2, 2, 0, 2, 2, 1, 1, 0, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1,
             2, 2, 2, 0, 2, 1, 0, 1, 0, 0, 1, 2, 0, 1, 1, 0, 2, 2, 1, 0, 2, 1, 1, 0, 1, 1, 1, 0, 1, 2, 1, 1,
             1, 2, 2, 1, 1, 0, 0, 2, 0, 0, 2, 0, 0, 0, 1, 0, 1, 1, 2, 1, 1, 1, 1, 0, 1, 0, 1, 2, 1, 2, 1, 1,
             0, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 2, 1, 1, 0, 2, 0, 2, 2, 2, 1, 2, 1, 1, 1, 1, 0, 1, 1,
             0, 2, 1, 1, 1, 1, 0, 2, 0, 0, 2, 1, 1, 2, 0, 2, 1, 1, 1, 0, 1, 1, 0, 2, 2, 1, 1, 0, 2, 2, 1, 2,
             0, 2, 0, 2, 1, 1, 1, 2, 0, 1, 0, 0, 2, 1, 2, 2, 2, 0, 1, 0, 0, 2, 2, 0, 1, 1, 2, 0, 0, 2, 0, 0,
             0, 0, 0, 1, 0, 2, 0, 1, 2, 0, 2, 1, 0, 2, 2, 2, 2, 0, 2, 0, 2, 2, 1, 2, 2, 2, 1, 2, 0, 1, 2, 2,
             2, 2, 1, 0, 2, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 2, 0, 2, 2, 0, 1, 1, 0, 1, 2, 1, 1, 2, 2, 1,
             1, 0, 0, 2, 0, 2, 1, 1, 0, 0, 1, 1, 1, 1, 2, 0, 0, 0, 2, 0, 0, 1, 2, 1, 1, 2, 0, 0, 0, 2, 0, 1,
             0, 1, 1, 2, 0, 1, 2, 2, 1, 1, 1, 2, 0, 1, 0, 2, 1, 2, 2, 1, 0, 1, 2, 1, 2, 0, 0, 0, 1, 2, 0, 2,
             0, 0, 1, 2, 0, 1, 2, 2, 0, 2, 2, 0, 0, 1, 0, 0, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 0, 0, 1, 2, 0, 2,
             1, 2, 1, 2, 1, 0, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 0, 0, 0, 2, 1, 2, 0, 2, 0, 1, 1, 2, 1,
             1, 2, 1, 1, 2, 1, 0, 2, 2, 1, 1, 1, 2, 2, 2, 0, 1, 0, 2, 1, 2, 1, 2, 2, 1, 1, 0, 2, 0, 1, 1, 0,
             0, 1, 2, 0, 0, 0, 0, 0, 2, 0, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 0, 2, 2, 1, 2, 0, 2, 2, 2, 2,
             2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 2, 2, 0, 2, 2, 1, 2, 0, 2, 2, 1, 2, 0, 1, 1, 1, 0, 1, 0,
             0, 0, 0, 2, 1, 0, 2, 2, 1, 2, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 2, 2, 1, 0, 2, 2, 0,
             0, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 0, 1, 1, 0, 0, 2, 0, 0, 0, 2, 0, 2, 1, 1, 0, 1, 0, 2, 0, 1, 0,
             1, 0, 1, 2, 0, 1, 2, 1, 1, 0, 1, 2, 1, 0, 0, 1, 0, 2, 2, 2, 1, 2, 0, 1, 0, 0, 2, 1, 0, 0, 2, 2,
             0, 2, 1, 2, 1, 1, 2, 2, 0, 2, 0, 1, 2, 1, 0, 1, 1, 2, 2, 0, 1, 1, 1, 0, 2, 2, 0, 1, 0, 1, 0, 2,
             2, 0, 2, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 2, 0, 1, 2, 2, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0,
             1, 1, 2, 0, 1, 0, 1, 1, 2, 1, 1, 2, 0, 2, 1, 2, 0, 1, 1, 2, 0, 2, 0, 2, 2, 2, 2, 0, 1, 0, 0, 2,
             2, 1, 1, 1, 2, 2, 0, 1, 2, 1, 0, 0, 1, 1, 0, 2, 0, 2, 1, 0, 2, 1, 1, 0, 0, 1, 2, 1, 2, 2, 1, 0,
             1, 2, 0, 1, 0, 2, 2, 1, 1, 0, 2, 1, 1, 1, 1, 1, 0, 0, 1, 1, 2, 1, 1, 2, 0, 0, 1, 1, 2, 2, 0, 1,
             2, 0, 1, 0, 2, 2, 0, 1, 2, 2, 2, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 1, 2, 0, 1, 0,
             0, 1, 2, 1, 0, 2, 2, 0, 0, 2, 1, 0, 0, 1, 1, 0, 0, 2, 1, 2, 0, 2, 1, 2, 2, 0, 0, 0, 2, 0, 2, 0,
             1, 2, 0, 0, 1, 1, 0, 1, 1, 2, 2, 1, 2, 0, 0]



def verification(feature, finalt):

    dissim = pd.read_csv('../reaperCSVs/cluster data/singledistancematrixto' + str(finalt) + '.csv')
    dissim.index = dissim['0Replay_id']
    del dissim['0Replay_id']
    matches = list(dissim.index)

    #mds = manifold.MDS(n_components=3, dissimilarity="precomputed", random_state=5, verbose=2)
    #results = mds.fit(dissim.values)
    #coords = pd.DataFrame(results.embedding_, index=dissim.index, columns=['Coord 1', 'Coord 2', 'Coord 3'])
    #Z = linkage(coords.values, 'ward')

    dffinal = pd.read_csv('../reaperCSVs/cluster data/cluster_data' + str(finalt) + '.csv')

    dffeature = pd.DataFrame(columns=range(720, finalt + 720, 720), index=matches)
    #dffeature['Names'] = fcluster(Z, 4, criterion='maxclust')
    dffeature['Names']=clustering
    #print(list(dffeature['Names'].values))

    for t in range(720, finalt + 720, 720):

        df = pd.read_csv('../reaperCSVs/cluster data/cluster_data' + str(t) + '.csv')
        df.index = df['0Replay_id']
        del df['0Replay_id']
        df = df.loc[df.index.isin(matches)]
        dffeature[t] = df[feature]

    dffeature.to_csv('../reaperCSVs/cluster data/verificationfile'+ feature + str(finalt) + '.csv')

    return dffeature


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


def readAndPrep(dir):
    df = pd.read_csv(dir)
    del df['Unnamed: 0']
    df = df.loc[:, (df!=0).any(axis=0)]
    return df


def clusterAtTimestamp(timestamp, ids):
    dir = '../data/Replays6-' + str(timestamp) + 's.csv'
    df = readAndPrep(dir)
    df, nameCol = rmName(df)
    df_km = cluster_KMeans(getPCs(df, 4), 3, True)

    labels = df_km['Names']
    dict = makeLabelDict(nameCol, labels)

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
            e2 = getPCs(e2, 3)
            #Methods.project_onto_R3(e, ['PC 1', 'PC 2', 'PC 3'])
            project_onto_R2(e2, ['PC 1', 'PC 2'], False)
            i += 1
    plt.show()
    return dfs_list


def rmName(df):
    if 'Name' in df.columns:
        nameCol = df['Name']
        dfNew = df.drop('Name', axis=1)
        return dfNew, nameCol
    else:
        return


def compare():

    f, axes = plt.subplots(2, 3)
    n = 0
    q = 60

    for i in range(60,150,30):

        s='../data/Replays2-'+str(i)+'.0s.csv'
        df = pd.read_csv(s)
        del df['Unnamed: 0']
        df = df.loc[:, (df != 0).any(axis=0)]

        df_pca=getPCs(df,2)
        df_pca = cluster_KMeans(df_pca, 2, True)

        names = list(set(df_pca.Names))
        i = 0

        for e in names:
            df_pca = df_pca.replace(e, i)
            i = i + 1

        axes[0,n].scatter(x=df_pca['PC 1'], y=df_pca['PC 2'], c=df_pca['Names'], cmap='rainbow')
        axes[0,n].set_title(str(q)+'s')
        n = n + 1
        q = 60 + 30 * n

    m = 0
    q = 150

    for i in range(150, 240, 30):

        s = '../data/Replays2-' + str(i) + '.0s.csv'
        df = pd.read_csv(s)
        del df['Unnamed: 0']
        df = df.loc[:, (df != 0).any(axis=0)]

        df_pca = getPCs(df, 2)
        df_pca = cluster_KMeans(df_pca, 2, True)

        names = list(set(df_pca.Names))
        i = 0
        for e in names:
            df_pca = df_pca.replace(e, i)
            i = i + 1

        axes[1, m].scatter(x=df_pca['PC 1'], y=df_pca['PC 2'], c=df_pca['Names'], cmap='rainbow')
        axes[1, m].set_title(str(q)+'s')
        m = m+1
        q = 150+30*m

    plt.show()


def project_onto_R3(df, cols):

    if 'Names' in df.columns:
        names = list(set(df.Names))
        i = 0
        for e in names:
            df = df.replace(e, i)
            i = i+1

        ax = plt.axes(projection='3d')
        ax.scatter3D(df[cols[0]], df[cols[1]], df[cols[2]], c=df['Names'], cmap='rainbow')

    else:
        ax = plt.axes(projection='3d')
        ax.scatter3D(df[cols[0]], df[cols[1]], df[cols[2]],'kx')


def project_onto_R2(df, cols, plot):
    if 'Names' in df.columns:
        names = list(set(df.Names))
        i = 0
        for e in names:
            df = df.replace(e, i)
            i = i + 1

        plt.scatter(x=df[cols[0]], y=df[cols[1]], c=df['Names'], cmap='rainbow')
    else:
        plt.scatter(x=df[cols[0]], y=df[cols[1]])

    if plot:
        plt.show()


def linkageType(df,type):

    dfNew, nameCol = rmName(df) #rms names

    if 'Names' in df.columns:
        data = df.drop('Names', axis=1)
        data = data.values
    else:
        data = df.values

    Z=linkage(data,type)
    dendrogram(Z, no_labels=True)
    plt.ylabel('Tolerance')
    plt.xlabel('Index in data')
    plt.title('Hierarchical dendogram;'+type+' linkage.')
    plt.show()


def heatMap(df):
    df = rmName(df)
    plt.figure()
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=plt.get_cmap('magma'),
                square=True)
    plt.show()
    return corr


def seabornHeatmap(df):

    df = rmName(df)
    if 'Names' in df.columns:
        df=df.drop('Names',axis=1)
        sns.clustermap(df,robust=True)
        plt.show()
    else:
        sns.clustermap(df)
        plt.show()


def parallelCoordinates(df):
    #df = rmName(df)
    plt.figure()
    plt.title('Parallel Coordinates plot')
    pd.plotting.parallel_coordinates(frame=df, class_column='Names', colormap=plt.get_cmap('tab10'))
    plt.show()


def radViz(df):
    df = rmName(df)
    plt.figure()
    plt.title('Radviz plot')
    pd.plotting.radviz(frame=df, class_column='Names', colormap=plt.get_cmap('tab10'))
    plt.show()


def parallelCoordinates_Clusters(df):
    df = rmName(df)
    clusters = set(list(df['Names']))
    columns = df.columns
    first = True
    clusterMean = []
    for e in clusters:
        cluster = df[df['Names'] == e]
        if first:
            first = False
            clusterMean = cluster[cluster.columns].mean().values
        else:
            clusterMean = np.vstack([clusterMean,cluster[cluster.columns].mean().values])
    plotDF = pd.DataFrame(clusterMean)
    plotDF['Names'] = clusters


    plt.title('Parallel Coordinates plot')
    pd.plotting.parallel_coordinates(frame=plotDF, class_column='Names', colormap=plt.get_cmap('tab10'))
    plt.grid(False)


def cluster_DBSCAN(df, eps, min_samples, keepOutliers, keepVarnames): #Hanterar dataframe
    df, nameCol = rmName(df)
    # init:
    labelsArray = []

    if 'Names' in df.columns:
        data = df.drop('Names', axis=1)
        data = data.values
    else:
        data = df.values


    X = StandardScaler().fit_transform(data)
    print('DBSCAN on ' + str(len(data[:, 1])) + ' points in ' + str(len(data[1, :])) + ' dimensions.')
    print('Clustering parameters set to eps=' + str(eps) + ', min_samples=' + str(min_samples) + '.')
    print()

    # Clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    for i in range(0,len(db.labels_)):
        if db.labels_[i]==-1:
            labelsArray.append('Outlier')
        else:
            labelsArray.append('Cluster '+str(db.labels_[i]+1))

    if n_clusters == 0:
        raise ValueError('No clusters found, change params.')
    print(str(n_clusters) + " clusters found.")
    print()

    if not keepVarnames:
        columns = ['Var %i' % i for i in range(1, len(data[1, :]) + 1)]
    else:
        if 'Names' in df.columns:
            df = df.drop('Names', axis=1)
        columns = df.columns

    dfNew=pd.DataFrame(data=data, columns=columns)

    dfNew['Names']=labelsArray

    print('#Points classified as outliers: ' + str(len(dfNew.loc[dfNew['Names'] == 'Outlier'])) + '.')
    for i in range(0, n_clusters, 1):
            print('#Points in cluster ' + str(i+1) + ': ' + str(len(dfNew.loc[dfNew['Names'] == 'Cluster '+str(i+1)]))+'.')
    dfNew['Name'] = nameCol
    if keepOutliers:
        return dfNew
    else:
        return dfNew.loc[dfNew['Names'] != 'Outlier']


def cluster_KMeans(df, k, keepVarnames): #Hanterar dataframe
    # init:
    #df, nameCol = rmName(df)
    labelsArray = []
    if 'Names' in df.columns:
        data = df.drop('Names', axis=1)
        data = data.values
    else:
        data = df.values

    X = StandardScaler().fit_transform(data)
    print('Executing K-Means clustering on ' + str(len(data[:, 0])) + ' points.')
    print('Looking for k=' + str(k) + ' clusters.')
    print()

    # Clustering
    km = KMeans(n_clusters=k, random_state=0, init = 'k-means++').fit(X)
    labels = km.labels_
    n_clusters = len(set(labels))
    print(str(n_clusters) + " clusters found.")

    for i in range(0,len(km.labels_)):
        labelsArray.append('Cluster '+str(km.labels_[i] + 1))

    if not keepVarnames:
        columns = ['Var %i' % i for i in range(1,len(data[1, :])+1)]
    else:
        if 'Names' in df.columns:
            df = df.drop('Names', axis=1)
        columns = df.columns

    dfNew = pd.DataFrame(data=data, columns=columns)

    dfNew['Names']=labelsArray

    for i in range(0, n_clusters, 1):
            print('#Points in cluster ' + str(i+1) + ': ' + str(len(dfNew.loc[dfNew['Names'] == 'Cluster '+str(i+1)]))+'.')
    #dfNew['Name'] = nameCol
    return dfNew


def elbowMethod(df, ks):
    df = rmName(df)
    distorsions = []

    for k in range(1, ks+1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)
        distorsions.append(kmeans.inertia_)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(1, ks+1), distorsions)
    plt.grid(True)
    plt.title('Elbow curve')
    plt.show()


def cluster_Hierarchical(df, k, linkageType, keepVarnames):
    df, nameCol = rmName(df)
    labelsArray = []

    if 'Names' in df.columns:
        data = df.drop('Names', axis=1)
        data = data.values
    else:
        data = df.values

    X = StandardScaler().fit_transform(data)
    print('Executing Agglomerative Hierarchical clustering on ' + str(len(data[:, 1])) + ' points.')
    print('Looking for k=' + str(k) + ' clusters.')
    print()

    # Clustering
    ac = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage=linkageType).fit(X)
    labels = ac.labels_
    n_clusters = len(set(labels))

    for i in range(0, len(ac.labels_)):
        labelsArray.append('Cluster ' + str(ac.labels_[i] + 1))

    if not keepVarnames:
        columns = ['Var %i' % i for i in range(1, len(data[1, :]) + 1)]
    else:
        if 'Names' in df.columns:
            df = df.drop('Names', axis=1)
        columns = df.columns

    dfNew = pd.DataFrame(data=data, columns=columns)

    dfNew['Names'] = labelsArray

    for i in range(0, n_clusters, 1):
        print('#Points in cluster ' + str(i + 1) + ': ' + str(
            len(dfNew.loc[dfNew['Names'] == 'Cluster ' + str(i + 1)])) + '.')
    dfNew['Name'] = nameCol
    return dfNew


def getPCs(df, n_components):
    #df, nameCol = rmName(df)
    if 'Names' in df.columns:
        data = df.drop('Names', axis=1)
    else:
        data = df

    tmp = data.values
    standard = StandardScaler()
    tmpS = standard.fit_transform(tmp)
    data = pd.DataFrame(tmpS)

    pca = PCA(n_components=n_components)
    pca.fit(data)
    columns = ['PC %i' % i for i in range(1,n_components+1)]
    df_pca = pd.DataFrame(pca.transform(data), columns=columns, index=df.index)

    if 'Names' in df.columns: #tror inte denna ifsats behövs..
        df_pca['Names'] = df['Names']
    #df_pca['Name'] = nameCol

    return df_pca


def clusterSparsePCA(df, n_components):
    df, nameCol = rmName(df)
    if 'Names' in df.columns:
        data = df.drop('Names', axis=1)
    else:
        data=df

    Data_Array = data.values
    standard = StandardScaler()
    Data_SArray = standard.fit_transform(Data_Array)
    data = pd.DataFrame(Data_SArray)

    pca = SparsePCA(n_components=n_components,normalize_components=True)
    pca.fit(data)
    columns = ['PC %i' % i for i in range(1,n_components+1)]
    df_pca = pd.DataFrame(pca.transform(data), columns=columns, index=df.index)

    if 'Names' in df.columns:
        df_pca['Names']=df['Names']

    df_pca['Name'] = nameCol
    return df_pca


def inversePCA(df):

    df, nameCol = rmName(df)
    if 'Names' in df.columns:
        df = df.drop('Names', axis=1)

    pca=PCA().fit(df)
    print('Number of components required to explain 95% of all variance: '+str(pca.n_components_))
    components = pca.transform(df)

    dfNew = pd.DataFrame(data=pca.inverse_transform(components))
    dfNew['Name'] = nameCol
    return dfNew


def explainedVariance(df):

    df = rmName(df)
    if 'Names' in df.columns:
        df = df.drop('Names', axis=1)

    pca = PCA().fit(df)

    print(np.cumsum(pca.explained_variance_ratio_))

    df = pd.DataFrame({'var': pca.explained_variance_ratio_,
                       'PC': ['PC %i' % i for i in range(1,len(df.columns)+1)]})

    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.bar(df['PC'],df['var'])
    ax1.set_xlabel('PC')
    ax1.set_ylabel('Explained variance')

    ax2.plot(np.cumsum(pca.explained_variance_ratio_))
    ax2.set_xlabel('Number of components')
    ax2.set_ylabel('Cumulative explained variance')
    plt.show()


def binaryCluster(df):

    if 'Names' not in df.columns:
        raise ValueError('Data not clustered.')

    df_dummies = pd.get_dummies(df['Names'])
    df_new = pd.concat([df, df_dummies], axis=1)
    del df_new['Names']

    return df_new


def pointBiserial(df, cols):

    #df = rmName(df)
    df = binaryCluster(df)

    if not all(elem in df.columns for elem in cols):
        raise ValueError('Dummy variable ' + np.setdiff1d(cols, df.columns) + ' not in DataFrame.')

    df = df.loc[:, (df != 0).any(axis=0)]

    for i in range(0, len(cols)):
        df[cols[i]].loc[df[cols[i]] == 1] = True
        df[cols[i]].loc[df[cols[i]] == 0] = False

    corr = pd.DataFrame()

    for c in cols:
        tmpCol = []
        for e in df.columns:
            tmp = pointbiserialr(df[c].values, df[e].values)
            tmpCol.append(tmp[0])
        corr[c] = tmpCol

    corr.index = df.columns
    corr = corr.T
    corr = corr.loc[:, (abs(corr) > 0.15).any(axis=0)]

    corr.drop(cols,axis=1)

    plt.figure()
    sns.set(font_scale=0.5)
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=plt.get_cmap('rainbow'), square=True, xticklabels=1)
    plt.show()

    return corr


def hopkins(X): #hittad på: https://matevzkunaver.wordpress.com/2017/06/20/hopkins-test-for-cluster-tendency/

    d = X.shape[1]
    # d = len(vars) # columns
    n = len(X)  # rows
    m = int(0.1 * n)  # heuristic from article [1]
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

    rand_X = sample(range(0, n, 1), m)

    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X, axis=0), np.amax(X, axis=0), d).reshape(1, -1), 2,
                                    return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])

    H = sum(ujd) / (sum(ujd) + sum(wjd))

    if isnan(H):
        print(ujd, wjd)
        H = 0

    return H


def hopkins_df(df): #hittad på: https://matevzkunaver.wordpress.com/2017/06/20/hopkins-test-for-cluster-tendency/

    if rmName(df) != False:
        df, nameCol = rmName(df)
    if 'Names' in df.columns:
        del df['Names']
    if 'Unnamed: 0' in df.columns:
        del df['Unnamed: 0']
    df = df.loc[:, (df != 0).any(axis=0)]
    X = df.values
    d = X.shape[1]
    # d = len(vars) # columns
    n = len(X)  # rows
    m = int(0.1 * n)  # heuristic from article [1]
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

    rand_X = sample(range(0, n, 1), m)

    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X, axis=0), np.amax(X, axis=0), d).reshape(1, -1), 2,
                                    return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])

    H = sum(ujd) / (sum(ujd) + sum(wjd))

    if isnan(H):
        print(ujd, wjd)
        H = 0

    return H

def dissimtosim(df):
    maxval=np.max(df.values)
    minval=np.min(df.values)
    max = pd.DataFrame([[maxval for col in range(len(df.columns))] for row in range(len(df.index))],
                       index=df.index, columns=df.columns)
    min = pd.DataFrame([[minval for col in range(len(df.columns))] for row in range(len(df.index))],
                       index=df.index, columns=df.columns)
    return max-df+min


def removedoubles(df):

    return df[~df.index.duplicated(keep='first')]


def dunnindexavg(df, dissim):

    clusters = list(set(df['Names']))
    inside = list()

    for c in clusters:

        intracl = list(set(df.loc[df['Names'] == c].index))
        dissimn = ssd.squareform(dissim[intracl].loc[intracl].values)
        inside.append(np.average(dissimn))

    visited = list()
    between = list()

    for c in clusters:

        visited.append(c)
        intracl = list(set(df.loc[df['Names'] == c].index))

        for c2 in clusters:

            if c2 not in visited:

                intracl2 = list(set(df.loc[df['Names'] == c2].index))
                dissimn = dissim[intracl].loc[intracl2]
                between.append(np.average(dissimn.values))

    return np.average(inside)/np.average(between)


def dunnindex(df, dissim):

    clusters = list(set(df['Names']))
    inside = list()

    for c in clusters:

        intracl = list(set(df.loc[df['Names'] == c].index))
        dissimn = ssd.squareform(dissim[intracl].loc[intracl].values)
        inside.append(np.average(dissimn))

    visited = list()
    between = list()

    for c in clusters:

        visited.append(c)
        intracl = list(set(df.loc[df['Names'] == c].index))

        for c2 in clusters:

            if c2 not in visited:

                intracl2 = list(set(df.loc[df['Names'] == c2].index))
                dissimn = dissim[intracl].loc[intracl2]
                between.append(np.average(dissimn.values))

    return np.max(inside)/np.min(between)


def overtime(time):

    dffirst = pd.read_csv('../data/Replays6-600s.csv')
    matches = list(set(dffirst['Name']))

    dffirst.index = dffirst['Name']
    dffirst = removedoubles(dffirst)

    del dffirst['Name']
    del dffirst['Unnamed: 0']

    df60 = pd.read_csv('../data/Replays6-60s.csv')
    df60.index = df60['Name']
    df60 = removedoubles(df60)

    matches2 = list(set(df60['Name']))

    for m in matches:

        if not (m in matches2):

            matches.remove(m)

    df60 = df60[df60['Name'].isin(matches)]

    del df60['Name']
    del df60['Unnamed: 0']

    dissimilarity = pd.DataFrame(data=distance_matrix(df60.values, df60.values, p=1), columns=df60.index, index=df60.index)

    for t in range(90, time+30, 30):

        dftmp = pd.read_csv('../data/Replays6-'+str(t)+'s.csv')
        matches2 = list(set(dftmp['Name']))

        for m in matches:

            if not (m in matches2):

                dissimilarity = dissimilarity.drop(m, axis=0)
                dissimilarity = dissimilarity.drop(m, axis=1)
                matches.remove(m)

        dftmp = dftmp[dftmp['Name'].isin(matches)]
        dftmp.index = dftmp['Name']
        dftmp = removedoubles(dftmp)

        del dftmp['Name']
        del dftmp['Unnamed: 0']

        d = pd.DataFrame(distance_matrix(dftmp.values, dftmp.values, p=1), index=dissimilarity.index, columns=dissimilarity.columns)

        values = d.values + dissimilarity.values

        dissimilarity = pd.DataFrame(data=values, columns=dissimilarity.columns, index=dissimilarity.columns)

        dissimilarity.to_csv('../data/newdissimilaritymatrixto'+str(t)+'.csv', encoding='utf-8', index=True)

        print('t='+str(t)+' completed', end='\r')

    return dissimilarity


def overtime2():

    df60 = pd.read_csv('../data/Replays6-60s.csv')
    df60.index = df60['Name']
    df60 = removedoubles(df60)
    df60 = df60.drop(['Name', 'Unnamed: 0'], axis=1)
    matches = list(df60.index)

    dissimilarity = pd.DataFrame(distance_matrix(df60.values, df60.values, p=1),
                                 columns=df60.index, index=df60.index)
    dissimilarity.to_csv('../data/new2dissimilaritymatrixto60.csv', encoding='utf-8', index=True)

    for t in range(90, 600+30, 30):

        dftmp = pd.read_csv('../data/Replays6-'+str(t)+'s.csv')

        dftmp.index = dftmp['Name']
        dftmp = removedoubles(dftmp)
        dftmp = dftmp.drop(['Name', 'Unnamed: 0'], axis=1)

        matches = [m for m in matches if m in list(dftmp.index)]

        dftmp = dftmp.loc[matches]
        dissimilarity = dissimilarity[matches].loc[matches]

        dissimilarity = pd.DataFrame(data=distance_matrix(dftmp.values, dftmp.values, p=1) + dissimilarity.values,
                                     columns=dissimilarity.columns, index=dissimilarity.index)

        dissimilarity.to_csv('../data/new2dissimilaritymatrixto'+str(t)+'.csv', encoding='utf-8', index=True)

        print('t='+str(t)+' completed', end='\r')

    return dissimilarity


def makeCompatible(df,dissim):

    dissim.index = dissim['Name']
    del dissim['Name']

    df = df.loc[df['Name'].isin(dissim.index)]
    df.index = df['Name']
    df = removedoubles(df)
    df = df.drop(['Name', 'Unnamed: 0'], axis=1)

    matches = [m for m in dissim.index if m in df.index]

    dissim = dissim[matches].loc[matches]

    return df, dissim


def checkOptimalClustering(t, expert):

    dissim = pd.read_csv('../data/new2dissimilaritymatrixto' + str(t) + '.csv')
    df = pd.read_csv('../data/Replays6-' + str(t) + 's.csv')

    df, dissim = makeCompatible(df, dissim)

    mindists = list()
    cls = list()

    Z = linkage(ssd.squareform(dissim.values), method='ward')

    for i in range(2, int(1/expert)+1):

        cl = fcluster(Z, i, criterion='maxclust')
        df['Names'] = cl

        if all(i >= expert*len(df.index) for i in [len(df.loc[df['Names'] == c].index) for c in list(set(cl))]):

            mindists.append(dunnindex(df, dissim))
            cls.append(i)

        print('t='+str(t)+': '+str(i)+'/'+str(int(1/expert))+' clusters checked.')

    return cls, mindists


def pointBiserialovertime():

    cls = [4, 2, 2, 3, 2, 5, 3, 4, 6, 4, 6, 4, 7, 6, 7, 3, 6, 6]

    times = [90, 120, 210,
             240, 270, 330,
             360, 390, 420,
             450, 480, 510,
             540, 570, 600]

    '''times = list()
    cls = list()
    minds=list()

    for t in range(90, 630, 30):

        clsopti, mind = checkOptimalClustering(t, expert)
        cls.append(clsopti[mind.index(np.min(mind))])
        minds.append(np.min(mind))
        times.append(t)

    print(cls)
    print(minds)'''

    j = 0

    for i in times:

        dissim = pd.read_csv('../data/newdissimilaritymatrixto600.csv')
        dissim.index = dissim['Name']
        matches = list(set(dissim['Name']))
        del dissim['Name']

        df = pd.read_csv('../data/Replays6-' + str(i) + 's.csv')
        df = df.loc[df['Name'].isin(matches)]
        df.index = df['Name']
        df = removedoubles(df)

        del df['Name']
        del df['Unnamed: 0']

        distArray = ssd.squareform(dissim.values)
        Z = linkage(distArray, method='ward')
        cl = fcluster(Z, 6, criterion='maxclust')

        df['Names'] = cl

        pointBiserial(df, [q for q in range(1, 7)])
        j = j + 1
        plt.show()


def parallellovertime():

    cls = [4, 2, 2,
           3, 2, 5,
           3, 4, 6,
           4, 6, 4,
           7, 6, 7,
           3, 6, 6]

    cls = [2, 2, 2,
           2, 3, 2,
           2, 2, 3,
           2, 3, 3,
           3, 4, 6]

    times = [90, 120, 210,
             240, 270, 330,
             360, 390, 420,
             450, 480, 510,
             540, 570, 600]

    '''times = list()
    cls = list()
    minds=list()

    for t in range(90, 630, 30):

        clsopti, mind = checkOptimalClustering(t, expert)
        cls.append(clsopti[mind.index(np.min(mind))])
        minds.append(np.min(mind))
        times.append(t)

    print(cls)
    print(minds)'''

    j = 0

    for i in times:

        print('t = ' + str(i)+'s.', end='\r')

        dissim = pd.read_csv('../data/newdissimilaritymatrixto600.csv')
        dissim.index = dissim['Name']
        matches = list(set(dissim['Name']))
        del dissim['Name']

        df = pd.read_csv('../data/Replays6-' + str(i) + 's.csv')
        df = df.loc[df['Name'].isin(matches)]
        df.index = df['Name']
        df = removedoubles(df)

        del df['Name']
        del df['Unnamed: 0']

        distArray = ssd.squareform(dissim.values)
        Z = linkage(distArray, method='ward')
        cl = fcluster(Z, 6, criterion='maxclust')
        df['Names'] = cl

        means = df.mean(axis=0)
        means = means.loc[means != 0]
        sds = df.std(axis=0)
        sds = sds.loc[sds != 0]

        sizes = list()
        parallell = pd.DataFrame()

        for cl in list(set(cl)):

            sizes.append(len(df.loc[df['Names'] == cl]))
            dftmp = df.loc[df['Names'] == cl]
            dftmp = dftmp.loc[:, (dftmp != 0).any(axis=0)]
            dftmp = dftmp.mean(axis=0)

            for f in dftmp.index:

                mean = means.loc[means.index == f]
                sd = sds.loc[sds.index == f]
                dftmp[f] = (dftmp.loc[dftmp.index == f]-mean)/sd

            parallell[cl] = dftmp

        print(sizes)
        parallell = parallell.T
        parallell['Names'] = [('Cluster '+str(q))+', size = '+str(sizes[q-1]) for q in range(1, 7)]
        parallell.index = parallell['Names']
        parallelCoordinates(parallell)
        plt.show()

        j = j + 1


def projectOptimalClustering(t):

    cls = [4, 2, 2,
           3, 2, 5,
           3, 4, 6,
           4, 6, 4,
           7, 6, 7,
           3, 6, 6]

    dissim = pd.read_csv('../data/newdissimilaritymatrixto570.csv')
    dissim.index = dissim['Name']
    matches = list(set(dissim['Name']))
    del dissim['Name']

    df = pd.read_csv('../data/Replays6-'+str(t)+'s.csv')
    df = df.loc[df['Name'].isin(matches)]
    df.index = df['Name']
    df = removedoubles(df)
    df = df.drop(['Name', 'Unnamed: 0'], axis=1)

    Z = linkage(ssd.squareform(dissim.values), method='ward')
    #cl = fcluster(Z, cls[int(t/30)-3], criterion='maxclust')
    cl = fcluster(Z, 6, criterion='maxclust')

    df['Names'] = cl
    df = getPCs(df, 3)
    project_onto_R3(df, ['PC ' + str(i) for i in range(1, 4)])

    plt.show()

