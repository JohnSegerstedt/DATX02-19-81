from clustering.Utilities import Classes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from pandas.plotting import parallel_coordinates
import seaborn as sns


def cluster_DBSCAN2(df,eps,min_samples,keepOutliers,keepVarnames): #Hanterar dataframe
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
       columns = []
       for i in range(0, len(data[1, :])):
            columns.append('Var ' + str(i+1))
    else:
        columns=df.columns

    dfNew=pd.DataFrame(data=data, columns=columns)
    print(dfNew)

    dfNew['Names']=labelsArray

    for i in range(0, n_clusters, 1):
        if i == 0 and keepOutliers:
            print('#Points classified as outliers: ' + str(len(dfNew.loc[dfNew['Names'] == 'Outlier'])) + '.')
        else:
            print('#Points in cluster ' + str(i+1) + ': ' + str(len(dfNew.loc[dfNew['Names'] == 'Cluster '+str(i+1)]))+'.')

    if keepOutliers:
        return dfNew
    else:
        return dfNew.loc[dfNew['Names'] != 'Outlier']



def cluster_KMeans2(df,k): #Hanterar dataframe
    # init:
    labelsArray = []
    if 'Names' in df.columns:
        data = df.drop('Names', axis=1)
        data = data.values
    else:
        data = df.values

    X = StandardScaler().fit_transform(data)
    print('Executing K-Means clustering on ' + str(len(data[:, 1])) + ' points.')
    print('Looking for k=' + str(k) + ' clusters.')
    print()

    # Clustering
    km = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = km.labels_
    n_clusters = len(set(labels))
    print(str(n_clusters) + " clusters found.")

    for i in range(0,len(km.labels_)):
        labelsArray.append('Cluster '+str(km.labels_[i]+1))

    if n_clusters == 0:
        raise ValueError('No clusters found, change params.')
    print(str(n_clusters) + " clusters found.")
    print()

    columns = []
    for i in range(0, len(data[1, :])):
        columns.append('Var ' + str(i+1))

    dfNew=pd.DataFrame(data=data, columns=columns)
    dfNew['Names']=labelsArray

    for i in range(0, n_clusters, 1):
        print('#Points in cluster ' + str(i+1) + ': ' + str(len(dfNew.loc[dfNew['Names'] == 'Cluster '+str(i+1)]))+'.')

    return dfNew


def cluster_Hierarchical(df,k,linkageType,keepOutliers,keepVarnames):
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
    print(str(n_clusters) + " clusters found.")

    for i in range(0, len(ac.labels_)):
        if ac.labels_[i] == -1:
            labelsArray.append('Outlier')
        else:
            labelsArray.append('Cluster ' + str(ac.labels_[i] + 1))

    if n_clusters == 0:
        raise ValueError('No clusters found, change params.')
    print(str(n_clusters) + " clusters found.")
    print()

    if not keepVarnames:
        columns = []
        for i in range(0, len(data[1, :])):
            columns.append('Var ' + str(i + 1))
    else:
        columns = df.columns

    dfNew = pd.DataFrame(data=data, columns=columns)
    print(dfNew)

    dfNew['Names'] = labelsArray

    for i in range(0, n_clusters, 1):
        if i == 0 and keepOutliers:
            print('#Points classified as outliers: ' + str(len(dfNew.loc[dfNew['Names'] == 'Outlier'])) + '.')
        else:
            print('#Points in cluster ' + str(i + 1) + ': ' + str(
                len(dfNew.loc[dfNew['Names'] == 'Cluster ' + str(i + 1)])) + '.')

    if keepOutliers:
        return dfNew
    else:
        return dfNew.loc[dfNew['Names'] != 'Outlier']


def wardLinkage(data):

    Z = linkage(data, 'ward')
    dn = dendrogram(Z)
    plt.ylabel('Tolerance')
    plt.xlabel('Index in data')
    plt.title('Hierarchical dendogram; ward linkage')
    plt.show()


def heatMap(df):
    plt.figure()
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=plt.get_cmap('RdGy'),
                square=True)
    plt.show()


def parallelCoordinates(df):
    plt.figure()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.title('Parallel Coordinates plot')
    pd.plotting.parallel_coordinates(
        frame=df, class_column='Names', colormap=plt.get_cmap('tab10'))
    plt.show()


def project_onto_R3(df, cols):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if {'Names'}.issubset(df):
        names = list(set(df.Names))
        groups = []
        for e in names:
            group = df[df['Names'] == e]
            group = group.drop('Names', axis=1)
            groups.append(group.values)
        for e in groups:
            ax.scatter(e[:, cols[0]], e[:, cols[1]], e[:, cols[2]])
    else:
        d = df.values
        ax.scatter(d[:,cols[0]], d[:,cols[1]], d[:,cols[2]])
    plt.show()


def getCluster(df, cluster_name): # Plockar ut cluster med namn 'cluster_name' ur en dataframe
    cluster=df.loc[df['Names'] == cluster_name]
    return cluster.drop('Names', axis=1)


def clusterPCA(df, cluster):
    group = df[df['Names'] == cluster]
    group = group.drop('Names', axis=1)
    data = group.values
    data=df.values
    pca = PCA(n_components=len(data[1,:])-1)
    pca.fit(data)

    return pca

