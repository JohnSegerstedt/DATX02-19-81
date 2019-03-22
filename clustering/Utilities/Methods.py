import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA, SparsePCA
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pointbiserialr
from pandas.plotting import parallel_coordinates
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan

'''
This method takes in an unlabeled DataFrame, and a dict with labels (replay -> label), and then assigns
labels corresponding to existing replays in the DataFrame. E.g. replay X is in Cluster 1, so this label is assigned.
Any replays that are not labeled after assignment are removed from the DataFrame.

@:param labelDict: dict (hashmap) with key = replay, and value = label
@:param df_unabeled: unlabeled DataFrame

@:returns: labeled DataFrame
'''
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


'''
Gets specific set of replays from DataFrame:

@:param df: DataFrame with replay data
@:param names: list with replay names

@returns: DataFrame with the replays that are both in df and namems param.
'''
def getReplays(df, names):
    return df.loc[df['Name'].isin(names)]


'''
Gets specific column from DataFrame

@:param df: DataFrame with Replay Data
@:param column: column name

@:returns: column from DataFrame
'''
def getColumn(df, column):
    return df.loc[df[column]]


'''
Makes labeldict from name- and label-arrays

@:param names: list with names
@:param labels: list with labels

@:returns: dict with key = name and value = label
'''
def makeLabelDict(names, labels):
    dict = {}
    for i in range(0, len(labels)):
        dict[names[i]] = labels[i]
    return dict


'''
Loads replay data and removes unnessecary columns from DataFrame

@:param dir: directory of data

@:returns: DataFrame with replay data
'''
def readAndPrep(dir):
    df = pd.read_csv(dir)
    del df['Unnamed: 0']
    df = df.loc[:, (df!=0).any(axis=0)]
    return df


'''
Removes Name column from dataframe if it is present

@:param df: DataFrame with replay data

@:return: dfNew: input DataFrame w/o "Name" column, nameCol: "Name" column
@:return: False if "Name" column is not in DataFrame
'''
def rmName(df):
    if 'Name' in df.columns:
        nameCol = df['Name']
        dfNew = df.drop('Name', axis=1)
        return dfNew, nameCol
    else:
        return False

'''
TODO
'''
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


'''
Projects DataFrame onto a three dimensional space and plots.

@:param df: DataFrame
@:param cols: columns on which the data is projected
'''
def project_onto_R3(df, cols):
    if rmName(df) != False:
        name = True
        df, nameCol = rmName(df)
    if 'Names' in df.columns:
        names = list(set(df.Names))
        i=0
        for e in names:
            df=df.replace(e,i)
            i=i+1
        ax = plt.axes(projection='3d')
        ax.scatter3D(df[cols[0]], df[cols[1]], df[cols[2]], c=df['Names'], cmap='rainbow')
    else:
        ax = plt.axes(projection='3d')
        ax.scatter3D(df[cols[0]], df[cols[1]], df[cols[2]],'kx')


'''
Projects DataFrame onto a two dimensional space and plots.

@:param df: DataFrame
@:param cols: columns on which the data is projected
@:param plot: boolean; if True: plot.show() is executed in method; if False: -||- is not -||-
'''
def project_onto_R2(df, cols, plot):
    if rmName(df) != False:
        name = True
        df, nameCol = rmName(df)
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


'''
Plots a hierarchical dendrogram from DataFrame Data. Type of Linkage can be chosen.

@:param df: DataFrame with data
@:param type: String specification of linkage type
'''
def linkageType(df,type):
    name = False
    if rmName(df) != False:
        name = True
        df, nameCol = rmName(df)

    if 'Names' in df.columns:
        data = df.drop('Names', axis=1)
        data = data.values
    else:
        data = df.values

    Z=linkage(data,type)
    dn = dendrogram(Z, no_labels=True)
    plt.ylabel('Tolerance')
    plt.xlabel('Index in data')
    plt.title('Hierarchical dendogram;'+type+' linkage.')
    plt.show()


'''
plots 2D heatmap, showing correlation between the columns in the input data

@:param df: input data (DataFrame)
'''
def heatMap(df):
    name = False
    if rmName(df) != False:
        name = True
        df, nameCol = rmName(df)
    plt.figure()
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=plt.get_cmap('magma'),
                square=True)
    plt.show()
    return corr


'''
TODO
'''
def seabornHeatmap(df):
    if rmName(df) != False:
        name = True
        df, nameCol = rmName(df)
    if 'Names' in df.columns:
        df=df.drop('Names',axis=1)
        sns.clustermap(df,robust=True)
        plt.show()
    else:
        sns.clustermap(df)
        plt.show()


'''
Plots Paralell Coordinates for labeled DataFrame (e.g each datapoint in DataFrame has label corresponding
to what cluster it belongs to)

@:param df: labeled DataFrame
'''
def parallelCoordinates(df):
    if rmName(df) != False:
        name = True
        df, nameCol = rmName(df)
    plt.figure()
    plt.title('Parallel Coordinates plot')
    pd.plotting.parallel_coordinates(frame=df, class_column='Names', colormap=plt.get_cmap('tab10'))
    plt.show()


'''
Creates radViz plot for labeled DataFrame (e.g each datapoint in DataFrame has label corresponding
to what cluster it belongs to)

@:param df: labeled DataFrame
'''
def radViz(df):
    if rmName(df) != False:
        name = True
        df, nameCol = rmName(df)
    plt.figure()
    plt.title('Radviz plot')
    pd.plotting.radviz(frame=df, class_column='Names', colormap=plt.get_cmap('tab10'))
    plt.show()


'''
Plots Paralell Coordinates for labeled DataFrame (e.g each datapoint in DataFrame has label corresponding
to what cluster it belongs to), where the data is simplified to one point per cluster (mean is taken in clusters)

@:param df: labeled DataFrame
'''
def parallelCoordinates_Clusters(df):
    if rmName(df) != False:
        name = True
        df, nameCol = rmName(df)
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


'''
Clusters data using Density Based Spatial Clustering with Applications to Noise, using SKLEARN implementation
Read more here: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
And here: http://www.cs.fsu.edu/~ackerman/CIS5930/notes/DBSCAN.pdf

@:param df: Input DataFrame with replay data
@:param eps: DBSCAN parameter; minimum distance to consider point as neighbouring
@:param min_samples: DBSCAN parameter; minimun points recuiered in neighbourhood of one point for that
point to be concidered a core point.
@:param keepOutliers: Boolean; True/False whether return data should contain points classified as outliers
@:param keepVarnames: Boolean; True/False whether return data should contain names of the variables in input data.

@:returns: DataFrame with new column "Names" where entry specifies the cluster a point belongs to//if it is a outlier.
'''
def cluster_DBSCAN(df, eps, min_samples, keepOutliers, keepVarnames): #Hanterar dataframe
    name = False
    if rmName(df) != False:
        name = True
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
    if name:
        dfNew['Name'] = nameCol
    if keepOutliers:
        return dfNew
    else:
        return dfNew.loc[dfNew['Names'] != 'Outlier']


'''
Clustering data using SKLEARN implementation of K-Means clustering.  
Read more here: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
And here: https://en.wikipedia.org/wiki/K-means_clustering

@:param df: Input DataFrame with replay data
@param k: Number of clusters k that are to be identified
@:param keepVarnames: Boolean; True/False whether return data should contain names of the variables in input data.

@:returns: DataFrame with new column "Names" where entry specifies the cluster a point belongs to//if it is a outlier.
'''
def cluster_KMeans(df, k, keepVarnames): #Hanterar dataframe
    # init:
    name = False
    if rmName(df) != False:
        name = True
        df, nameCol = rmName(df)
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
    if name:
        dfNew['Name'] = nameCol
    return dfNew



'''
Makes elbowPlot for DataFrame df with ks points; (essentially this plot tells us how much of the variance can be
explained as a function of number of clusters.).

@:param df: DataFrame with replay data
@:param ks: highest number of clusters to be considered for the plot
'''
def elbowMethod(df, ks):
    if rmName(df) != False:
        name = True
        df, nameCol = rmName(df)
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



'''
Clustering data using SKLEARN implementation of hierarchichal clustering with user-chosen linkage type.
Read more here: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
And here: https://en.wikipedia.org/wiki/Hierarchical_clustering

@:param df: Input DataFrame with replay data
@:param k: Number of clusters k that are to be identified
@:param linkageType: String specification of what linkage type is to be used.
@:param keepVarnames: Boolean; True/False whether return data should contain names of the variables in input data.

@:returns: DataFrame with new column "Names" where entry specifies the cluster a point belongs to//if it is a outlier.
'''
def cluster_Hierarchical(df, k, linkageType, keepVarnames):
    name = False
    if rmName(df) != False:
        name = True
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
    if name:
        dfNew['Name'] = nameCol
    return dfNew


'''
Calculates n Principal components of the input data, and returns the data represented with these
principal components as a basis.

@:param df: Input DataFrame
@param n_components: Number of principal components wanted

@returns: DataFrame with n_components columns, where the points are the input points represented in the basis
of the principal components identified.
'''
def getPCs(df, n_components):
    name = False
    if rmName(df) != False:
        name = True
        df, nameCol = rmName(df)
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
    if name:
        df_pca['Name'] = nameCol

    return df_pca


'''
TODO
'''
def clusterSparsePCA(df, n_components):
    name = False
    if rmName(df) != False:
        name = True
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

    if name:
        df_pca['Name'] = nameCol
    return df_pca


'''
TODO
'''
def inversePCA(df):
    name = False
    if rmName(df) != False:
        name = True
        df, nameCol = rmName(df)
    if 'Names' in df.columns:
        df = df.drop('Names', axis=1)

    pca=PCA().fit(df)
    print('Number of components required to explain 95% of all variance: '+str(pca.n_components_))
    components = pca.transform(df)

    dfNew = pd.DataFrame(data=pca.inverse_transform(components))
    if name:
        dfNew['Name'] = nameCol
    return dfNew



'''
Plots how much of the variance in the DataFrame is explained as a function of number of principal components

@:param df: Input DataFrame to be analyzed.
'''
def explainedVariance(df):
    if rmName(df) != False:
        name = True
        df, nameCol = rmName(df)
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


'''
TODO
'''
def binaryCluster(df):

    if 'Names' not in df.columns:
        raise ValueError('Data not clustered.')

    df_dummies = pd.get_dummies(df['Names'])
    df_new = pd.concat([df, df_dummies], axis=1)
    del df_new['Names']

    return df_new


'''
TODO
'''
def pointBiserial(df, cols):
    if rmName(df) != False:
        name = True
        df, nameCol = rmName(df)
    df = binaryCluster(df)

    for c in cols:
        if c not in df.columns:
            raise ValueError('Dummy variable ' + c + ' not in DataFrame.')

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

    for c in cols:
        del corr[c]

    plt.figure()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=plt.get_cmap('magma'), square=True)
    plt.show()

    return corr

'''
Calculates the probability that the data X contains clusters.
Read more here: https://en.wikipedia.org/wiki/Hopkins_statistic

@:param X: Input data

@:returns H: int in [0,1] corresponding to the probability that X has clusters using the Hopkins metric.
'''
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


'''
Calculates the probability that the data X contains clusters.
Read more here: https://en.wikipedia.org/wiki/Hopkins_statistic

@:param X: Input data (DataFrame)

@:returns H: int in [0,1] corresponding to the probability that X has clusters using the Hopkins metric.
'''
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
