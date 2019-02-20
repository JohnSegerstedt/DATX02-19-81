from clustering.Utilities import Classes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from pandas.plotting import parallel_coordinates
import seaborn as sns

def normalize(A):
    for i in range(0, len(A[1,:]), 1):
        if np.std(A[:,i]) != 0:
            A[:,i] = (A[:,i]-np.mean(A[:,i]))/np.std(A[:,i])
    return A


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


def cluster_DBSCAN(data, dim, eps, min_samples, outliers):
    #init:
    labelsArray = []
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
    for i in range(-1, n_clusters, 1):
        sets.append(Classes.Set(dim))
    for i in range(0,len(labels), 1):
        n_in_sets[labels[i]+1] += 1
        sets[labels[i]+1].__add__(Classes.Point(np.array([data[i,:]])))
    for i in range(0, len(n_in_sets), 1):
        if i == 0:
            print('#Points classified as outliers: ' + str(int(n_in_sets[0])) + '.')
        else:
            print('#Points in cluster ' + str(i) + ': ' + str(int(n_in_sets[i])))

    return setsToDataFrame(sets=sets, outliers=outliers)


def cluster_KMeans(data, dim, k):
    #init:
    labelsArray = []
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


def averageLinkage(data):
    Z = linkage(data, 'average')
    dn = dendrogram(Z)
    plt.ylabel('Tolerance')
    plt.xlabel('Index in data')
    plt.title('Hierarchical dendogram; average linkage')


def wardLinkage(data):
    Z = linkage(data, 'ward')
    dn = dendrogram(Z)
    plt.ylabel('Tolerance')
    plt.xlabel('Index in data')
    plt.title('Hierarchical dendogram; ward linkage')


def heatMap(data):
    df = pd.DataFrame()
    for i in range(0, len(data[1, :])):
        df['var' + str(i + 1)] = data[:, i]
    plt.figure()
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=plt.get_cmap('magma'),
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
    names = list(set(df.Names))
    groups = []
    for e in names:
        group = df[df['Names'] == e]
        groups.append(group.values)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_xlim([-20, 20])
    #ax.set_ylim([-20, 20])
    #ax.set_zlim([-20, 20])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    for e in groups:
        ax.scatter(e[:, cols[0]+1], e[:, cols[1]+1], e[:, cols[2]+1], 'kx')
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
