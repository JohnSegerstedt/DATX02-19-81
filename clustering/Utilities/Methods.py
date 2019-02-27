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


def project_onto_R3(df, cols):

   if 'Names' in df.columns:
       names = list(set(df.Names))
       i=0
       for e in names:
           df=df.replace(e,i)
           i=i+1
       ax = plt.axes(projection='3d')
       ax.scatter3D(df[cols[0]], df[cols[1]], df[cols[2]], c=df['Names'], cmap='rainbow')
       plt.show()
   else:
       ax = plt.axes(projection='3d')
       ax.scatter3D(df[cols[0]], df[cols[1]], df[cols[2]],'kx')
       plt.show()


def project_onto_R2(df, cols):
    if 'Names' in df.columns:
        names = list(set(df.Names))
        i = 0
        for e in names:
            df = df.replace(e, i)
            i = i + 1

        plt.scatter(x=df[cols[0]], y=df[cols[1]], c=df['Names'], cmap='rainbow')
        plt.show()
    else:
        plt.scatter(x=df[cols[0]], y=df[cols[1]])
        plt.show()


def linkageType(df,type):

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


def heatMap(df):
    #df = pd.DataFrame()
    #for i in range(0, len(data[1, :])):
    #    df['var' + str(i + 1)] = data[:, i]
    plt.figure()
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=plt.get_cmap('magma'),
                square=True)
    plt.show()
    return corr


def seabornHeatmap(df):
    if 'Names' in df.columns:
        df=df.drop('Names',axis=1)
        sns.clustermap(df,robust=True)
        plt.show()
    else:
        sns.clustermap(df)
        plt.show()


def parallelCoordinates(df):

    plt.figure()
    plt.title('Parallel Coordinates plot')
    pd.plotting.parallel_coordinates(frame=df, class_column='Names', colormap=plt.get_cmap('tab10'))
    plt.show()


def cluster_DBSCAN(df,eps,min_samples,keepOutliers,keepVarnames): #Hanterar dataframe
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

    for i in range(0, n_clusters, 1):
        if i == 0 and keepOutliers:
            print('#Points classified as outliers: ' + str(len(dfNew.loc[dfNew['Names'] == 'Outlier'])) + '.')
        else:
            print('#Points in cluster ' + str(i+1) + ': ' + str(len(dfNew.loc[dfNew['Names'] == 'Cluster '+str(i+1)]))+'.')

    if keepOutliers:
        return dfNew
    else:
        return dfNew.loc[dfNew['Names'] != 'Outlier']


def cluster_KMeans(df,k,keepVarnames): #Hanterar dataframe
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
    km = KMeans(n_clusters=k,random_state=0).fit(X)
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

    return dfNew


def cluster_Hierarchical(df,k,linkageType,keepVarnames):

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

    return dfNew


def clusterPCA(df,n_components):

    if 'Names' in df.columns:
        data = df.drop('Names', axis=1)
    else:
        data=df

    Data_Array = data.values
    standard = StandardScaler()
    Data_SArray = standard.fit_transform(Data_Array)
    data = pd.DataFrame(Data_SArray)

    pca = PCA(n_components = n_components)
    pca.fit(data)
    columns = ['PCA %i' % i for i in range(1,n_components+1)]
    df_pca = pd.DataFrame(pca.transform(data), columns=columns, index=df.index)

    if 'Names' in df.columns:
        df_pca['Names']=df['Names']

    return df_pca


def clusterSparsePCA(df,n_components):

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
    columns = ['PCA %i' % i for i in range(1,n_components+1)]
    df_pca = pd.DataFrame(pca.transform(data), columns=columns, index=df.index)

    if 'Names' in df.columns:
        df_pca['Names']=df['Names']

    return df_pca


def inversePCA(df):

    if 'Names' in df.columns:
        df = df.drop('Names', axis=1)

    pca=PCA().fit(df)
    print('Number of components required to explain 95% of all variance: '+str(pca.n_components_))
    components = pca.transform(df)
    return pd.DataFrame(data=pca.inverse_transform(components))


def explainedVariance(df):

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


def pointBiserial(df, cluster):

    if cluster not in df.columns:
        raise ValueError('No dummy variable.')

    corr=pd.DataFrame()

    df[cluster].loc[df[cluster] == 1] = True
    df[cluster].loc[df[cluster] == 0] = False

    for e in df.columns:
        corr[e] = pointbiserialr(df[cluster].values, df[e].values)

    corr = corr.drop(corr.index[1])

    plt.figure()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=plt.get_cmap('magma'), square=True)
    plt.show()

    return corr
