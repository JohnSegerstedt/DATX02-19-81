import matplotlib.pyplot as plt
import pandas as pd
from clustering.Utilities import Methods
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import manifold
from sklearn.cluster import SpectralClustering
import scipy.spatial.distance as ssd

distancemetric = 'cosine'   #Vilket distansmått man ska använda
hierarcmetric = 'ward'      #Vilken metod den hierarkiska klustringen ska använda sig av
finalt = 12960              #Hur sent in i matchen man vill kolla (i frames)
nrofpoints = 4000           #Hur många matcher man vill kolla på (ju fler desto segare)
recompute = True            #Om distansmatriserna ska räknas om
scale = True                #Om datan ska skalas med MinMax vid omräkning av distansmatriserna

removecols = ['0P1_result','0P2_result','Unnamed: 0', '0Frame_id', '0P1_mmr',
              '0P2_mmr','0Replay_id', 'P1_Supply_army', 'P1_Supply_total',
              'P1_Supply_used', 'P1_Supply_workers', 'P1_minerals', 'P1_vespene',
              'P2_Supply_army', 'P2_Supply_total', 'P2_Supply_used', 'P2_Supply_workers',
              'P2_minerals', 'P2_vespene']

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

if recompute:

    dffinal = pd.read_csv('../reaperCSVs/cluster data/cluster_data'+str(finalt)+'.csv')
    dffinal = dffinal.tail(nrofpoints)
    matchesfin = list(dffinal['0Replay_id'])

    df720 = pd.read_csv('../reaperCSVs/cluster data/cluster_data720.csv')
    df720.index = df720['0Replay_id']
    df720 = df720.drop(removecols, axis=1)
    df720 = df720.loc[df720.index.isin(matchesfin)]
    df720 = Methods.removedoubles(df720)

    matches = list(df720.index)

    if scale:
        dissimilarity = pd.DataFrame(np.around(squareform(pdist(MinMaxScaler().fit_transform(df720.values),
                                    distancemetric)), 3), columns=df720.index,
                                     index=df720.index)
    else:
        dissimilarity = pd.DataFrame(
            np.around(squareform(pdist(df720.values, distancemetric)), 3),
            columns=df720.index, index=df720.index)

    dissimilarity.to_csv('../reaperCSVs/cluster data/distancematrixto720.csv', encoding='utf-8', index=True)

    for t in range(720+720, finalt+720, 720):

        dftmp = pd.read_csv('../reaperCSVs/cluster data/cluster_data'+str(t)+'.csv')
        dftmp.index = dftmp['0Replay_id']
        dftmp = dftmp.drop(removecols, axis=1)

        dftmp = dftmp.loc[dftmp.index.isin(matchesfin)]
        dftmp = Methods.removedoubles(dftmp)

        dissimilarity = dissimilarity.drop([m for m in matchesfin if m not in list(dftmp.index)], axis=0)
        dissimilarity = dissimilarity.drop([m for m in matchesfin if m not in list(dftmp.index)], axis=1)

        matchesfin = list(dissimilarity.index)

        if scale:
            dissimilarity = pd.DataFrame(data=np.around(squareform(pdist(MinMaxScaler().fit_transform(dftmp.values), distancemetric)) + dissimilarity.values, 3),
                                             columns=dissimilarity.columns, index=dissimilarity.index)
        else:
            dissimilarity = pd.DataFrame(data=np.around(squareform(pdist(dftmp.values, distancemetric)) + dissimilarity.values, 3),
                                         columns=dissimilarity.columns, index=dissimilarity.index)

        dissimilarity.to_csv('../reaperCSVs/cluster data/distancematrixto'+str(t)+'.csv', encoding='utf-8', index=True)
        print(str(t)+'/'+str(finalt),end='\r')

dissim = pd.read_csv('../reaperCSVs/cluster data/distancematrixto'+str(finalt)+'.csv')
dissim.index = dissim['0Replay_id']
del dissim['0Replay_id']

sim = Methods.dissimtosim(dissim)

coords = pd.DataFrame(index=dissim.index)

print('Dunn index for spectral clustering:')
spect = list()

for i in range(2, 20):

    clustering = SpectralClustering(n_clusters=i, assign_labels="discretize", random_state=0,
                                    affinity='precomputed').fit(sim.values)

    coords['Names'] = clustering.labels_

    dunn = dunnindex(coords, dissim)
    spect.append(dunn)
    print(dunn)

print('Dunn index for hierarchical clustering:')

Z = linkage(squareform(dissim.values), hierarcmetric)
hier = list()

for i in range(2, 20):

    coords['Names'] = fcluster(Z, i, criterion='maxclust')

    dunn = dunnindex(coords, dissim)
    hier.append(dunn)
    print(dunn)

mds = manifold.MDS(n_components=3, dissimilarity="precomputed", random_state=5, verbose=2)
results = mds.fit(dissim.values)
coords = pd.DataFrame(results.embedding_, index=dissim.index, columns=['Coord 1', 'Coord 2', 'Coord 3'])

if np.min(hier) <= np.min(spect):
    print('Best candidate: hierarchical clustering with '+str(hier.index(np.min(hier))+2)+' clusters')
    Z = linkage(squareform(dissim.values), 'ward')
    coords['Names'] = fcluster(Z, hier.index(np.min(hier))+2, criterion='maxclust')

else:
    print('Best candidate: spcetral clustering with ' + str(spect.index(np.min(spect)) + 2) + ' clusters')
    clustering = SpectralClustering(n_clusters=spect.index(np.min(spect))+2, assign_labels="discretize",
                                    random_state=0, affinity='precomputed').fit(sim.values)
    coords['Names'] = clustering.labels_

project_onto_R3(coords, ['Coord 1', 'Coord 2', 'Coord 3'])
plt.show()