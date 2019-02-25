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
def normalize(df):

    if 'Names' in df.columns:
        data = df.drop('Names', axis=1)
        A = data.values
    else:
        A = df.values

    for i in range(0, len(A[1,:]), 1):
        if np.std(A[:,i]) != 0:
            A[:,i] = (A[:,i]-np.mean(A[:,i]))/np.std(A[:,i])

    dfNew=pd.DataFrame(data=A,columns=data.columns)
    dfNew['Names']=df.loc[:,'Names']

    return dfNew


def setsToDataFrame(sets, outliers):
    names = []
    empty = True
    clusterIndx = 1
    for i in range(0, len(sets), 1):
        if not sets[i].empty:
            for j in range(0, len(sets[i].pList), 1):
                if i == 0 and outliers:
                    names.append('Outliers')
                else:
                    names.append('Cluster ' + str(clusterIndx))
                if empty:
                    positions = sets[i].posMat[j, :]
                    empty = False
                else:
                    positions = np.vstack([positions, sets[i].posMat[j, :]])
        clusterIndx += 1
    positions = normalize(positions)
    df = pd.DataFrame({'Names':names})
    for i in range(0, len(positions[1,:])):
        df['var' + str(i+1)] = positions[:,i]
    return df


def cluster_DBSCAN(df, dim, eps, min_samples,keepOutliers):
    #init:
    labelsArray = []

    if 'Names' in df.columns:
        data = df.drop('Names', axis=1)
        data = data.values
    else:
        data = df.values

    X = StandardScaler().fit_transform(data)
    print('DBSCAN on ' + str(len(data[:,1])) + ' points in ' + str(dim) + ' dimensions.')
    print('Clustering parameters set to eps=' + str(eps) + ', min_samples=' + str(min_samples) + '.')
    print()

    #Clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    labelsArray.append(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_in_sets = np.zeros(n_clusters+1)

    if n_clusters == 0:
        raise ValueError('No clusters found, change params.')
    print(str(n_clusters) + " clusters found.")
    print()

    sets = []
    start = 0

    if keepOutliers:
        start = -1

    for i in range(start, n_clusters, 1):
        sets.append(Classes.Set(dim))

    for i in range(0,len(labels), 1):
        if not keepOutliers:
            if labels[i] != -1:
                n_in_sets[labels[i]] += 1
                sets[labels[i]].__add__(Classes.Point(np.array([data[i, :]])))
        else:
            n_in_sets[labels[i] + 1] += 1
            sets[labels[i] + 1].__add__(Classes.Point(np.array([data[i, :]])))

    for i in range(0, len(n_in_sets) -1 - start, 1):
        if i == 0 and keepOutliers:
            print('#Points classified as outliers: ' + str(int(n_in_sets[0])) + '.')
        else:
            print('#Points in cluster ' + str(i + 1 + start) + ': ' + str(int(n_in_sets[i])))

    return setsToDataFrame(sets=sets, outliers=keepOutliers)

def cluster_KMeans(df, dim, k):
    #init:
    labelsArray = []
    if 'Names' in df.columns:
        data = df.drop('Names', axis=1)
        data = data.values
    else:
        data = df.values
    X = StandardScaler().fit_transform(data)
    print('Executing K-Means clustering on ' + str(len(data[:,1])) + ' points.')
    print('Looking for k=' + str(k) + ' clusters.')
    print()

    #Clustering
    km = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = km.labels_
    labelsArray.append(labels)
    n_clusters = len(set(labels))
    print(str(n_clusters) + " clusters found.")


    #Extract sets
    n_in_sets = np.zeros(n_clusters)
    sets = []
    for i in range(0, n_clusters, 1):
        sets.append(Classes.Set(dim))
    for i in range(0,len(labels), 1):
        n_in_sets[labels[i]] += 1
        sets[labels[i]].__add__(Classes.Point(np.array([data[i,:]])))
    for i in range(0, len(n_in_sets), 1):
            print('#Points in cluster ' + str(i) + ': ' + str(int(n_in_sets[i])))

    return setsToDataFrame(sets=sets, outliers=False)

def linkageType(df, type):

    if 'Names' in df.columns:
        data = df.drop('Names', axis=1)
        data = data.values
    else:
        data = df.values

    if type == 'single':
        Z = linkage(data, 'single')
    elif type == 'complete':
        Z = linkage(data,'complete')
    elif type == 'average':
        Z = linkage(data,'average')
    elif type == 'ward':
        Z = linkage(data,'ward')
    else: raise ValueError('Unallowed type.')

    dn = dendrogram(Z,no_labels=True)
    plt.ylabel('Tolerance')
    plt.xlabel('Index in data')
    plt.title('Hierarchical dendogram;'+type+' linkage.')
    plt.show()


def singleLinkage(data):

    Z = linkage(data, 'single')
    dn = dendrogram(Z)
    plt.ylabel('Tolerance')
    plt.xlabel('Index in data')
    plt.title('Hierarchical dendogram; single linkage')


def completeLinkage(data):

    Z = linkage(data, 'complete')
    dn = dendrogram(Z)
    plt.ylabel('Tolerance')
    plt.xlabel('Index in data')
    plt.title('Hierarchical dendogram; complete linkage')


def averageLinkage(df):
    if 'Names' in df.columns:
        data = df.drop('Names', axis=1)
        data = data.values
    else:
        data = df.values
    Z = linkage(data, 'average')
    dn = dendrogram(Z)
    plt.ylabel('Tolerance')
    plt.xlabel('Index in data')
    plt.title('Hierarchical dendogram; average linkage')
    plt.show()



def generateUniformXYZ(x, y, z, xRng, yRng, zRng, n):
    xPts = x + xRng * (.5 - np.random.rand(n))
    yPts = y + yRng * (.5 - np.random.rand(n))
    zPts = z + zRng * (.5 - np.random.rand(n))
    return xPts, yPts, zPts


def initTmp():
    x1, y1, z1 = generateUniformXYZ(-5, 0, 0, 4, 4, 4, 100)
    x2, y2, z2 = generateUniformXYZ(10, 0, -4, 0, 20, 0, 0)
    x3, y3, z3 = generateUniformXYZ(10, 0, -4, 0, 0, 10, 100)
    x4, y4, z4 = generateUniformXYZ(10, 10, 10, 10, 10, 10, 0)
    x5, y5, z5 = generateUniformXYZ(0, 0, 0, 40, 40, 40, 1000)

    x = np.transpose(np.hstack([x1, x2, x3, x4, x5]))
    y = np.transpose(np.hstack([y1, y2, y3, y4, y5]))
    z = np.transpose(np.hstack([z1, z2, z3, z4, z5]))
    A = np.vstack([x, y, z])
    return A.T

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

    def normalize(df):

        if 'Names' in df.columns:
            data = df.drop('Names', axis=1)
            A = data.values
        else:
            A = df.values

        for i in range(0, len(A[1, :]), 1):
            if np.std(A[:, i]) != 0:
                A[:, i] = (A[:, i] - np.mean(A[:, i])) / np.std(A[:, i])

        dfNew = pd.DataFrame(data=A, columns=data.columns)
        dfNew['Names'] = df.loc[:, 'Names']

        return dfNew

    def setsToDataFrame(sets, outliers):
        names = []
        empty = True
        clusterIndx = 1
        for i in range(0, len(sets), 1):
            if not sets[i].empty:
                for j in range(0, len(sets[i].pList), 1):
                    if i == 0 and outliers:
                        names.append('Outliers')
                    else:
                        names.append('Cluster ' + str(clusterIndx))
                    if empty:
                        positions = sets[i].posMat[j, :]
                        empty = False
                    else:
                        positions = np.vstack([positions, sets[i].posMat[j, :]])
            clusterIndx += 1
        positions = normalize(positions)
        df = pd.DataFrame({'Names': names})
        for i in range(0, len(positions[1, :])):
            df['var' + str(i + 1)] = positions[:, i]
        return df


def cluster_DBSCAN(df, dim, eps, min_samples,keepOutliers):
    #init:
    labelsArray = []

    if 'Names' in df.columns:
        data = df.drop('Names', axis=1)
        data = data.values
    else:
        data = df.values

    X = StandardScaler().fit_transform(data)
    print('DBSCAN on ' + str(len(data[:,1])) + ' points in ' + str(dim) + ' dimensions.')
    print('Clustering parameters set to eps=' + str(eps) + ', min_samples=' + str(min_samples) + '.')
    print()

    #Clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    labelsArray.append(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_in_sets = np.zeros(n_clusters+1)

    if n_clusters == 0:
        raise ValueError('No clusters found, change params.')
    print(str(n_clusters) + " clusters found.")
    print()

    sets = []
    start = 0

    if keepOutliers:
        start = -1

    for i in range(start, n_clusters, 1):
        sets.append(Classes.Set(dim))

    for i in range(0,len(labels), 1):
        if not keepOutliers:
            if labels[i] != -1:
                n_in_sets[labels[i]] += 1
                sets[labels[i]].__add__(Classes.Point(np.array([data[i, :]])))
        else:
            n_in_sets[labels[i] + 1] += 1
            sets[labels[i] + 1].__add__(Classes.Point(np.array([data[i, :]])))

    for i in range(0, len(n_in_sets) -1 - start, 1):
        if i == 0 and keepOutliers:
            print('#Points classified as outliers: ' + str(int(n_in_sets[0])) + '.')
        else:
            print('#Points in cluster ' + str(i + 1 + start) + ': ' + str(int(n_in_sets[i])))

    return setsToDataFrame(sets=sets, outliers=keepOutliers)

def cluster_KMeans(df, dim, k):
    #init:
    labelsArray = []
    if 'Names' in df.columns:
        data = df.drop('Names', axis=1)
        data = data.values
    else:
        data = df.values
    X = StandardScaler().fit_transform(data)
    print('Executing K-Means clustering on ' + str(len(data[:,1])) + ' points.')
    print('Looking for k=' + str(k) + ' clusters.')
    print()

    #Clustering
    km = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = km.labels_
    labelsArray.append(labels)
    n_clusters = len(set(labels))
    print(str(n_clusters) + " clusters found.")


    #Extract sets
    n_in_sets = np.zeros(n_clusters)
    sets = []
    for i in range(0, n_clusters, 1):
        sets.append(Classes.Set(dim))
    for i in range(0,len(labels), 1):
        n_in_sets[labels[i]] += 1
        sets[labels[i]].__add__(Classes.Point(np.array([data[i,:]])))
    for i in range(0, len(n_in_sets), 1):
            print('#Points in cluster ' + str(i) + ': ' + str(int(n_in_sets[i])))

    return setsToDataFrame(sets=sets, outliers=False)




def wardLinkage(data):

    Z = linkage(data, 'ward')
    dn = dendrogram(Z)
    plt.ylabel('Tolerance')
    plt.xlabel('Index in data')
    plt.title('Hierarchical dendogram; ward linkage')


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



