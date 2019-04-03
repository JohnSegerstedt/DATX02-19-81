from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from pylab import figure
from sklearn import mixture, preprocessing, manifold
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from clustering.Utilities import Methods
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.spatial.distance as ssd
from scipy.spatial import distance_matrix
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import manifold
from random import sample
from sklearn.cluster import SpectralClustering


'''df = pd.read_csv('../reaperCSVs/cluster data/cluster_data7200.csv')
df = df.drop(removecols, axis=1)

for cols in list(df.columns):

    df = Methods.verification(cols, 12240)
    Methods.parallelCoordinates(df)
    plt.show()
    print(cols)

for cols in list(df.columns):

    df = pd.read_csv('../reaperCSVs/cluster data/verificationfile'+cols+'10800.csv')
    df.index = df['Unnamed: 0']
    del df['Unnamed: 0']
    #print(df)
    Methods.parallelCoordinates(df)
    plt.show()'''

def seriation(Z, N, cur_index): #Hittad på: https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return (seriation(Z, N, left) + seriation(Z, N, right))

def compute_serial_matrix(dist_mat, method="ward"): #Hittad på: https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist

'''dissim = pd.read_csv('../reaperCSVs/cluster data/distancematrixto7200.csv')
dissim.index = dissim['0Replay_id']
del dissim['0Replay_id']
dissim = dissim.tail(1000)
dissim = dissim.drop([m for m in dissim.columns if m not in dissim.index], axis=1)

dissimsorted = compute_serial_matrix(dissim.values)
#sns.heatmap(dissimsorted, vmin=20)
#plt.show()

mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=5, verbose=2)
results = mds.fit(dissim.values)
coords = pd.DataFrame(results.embedding_, index=dissim.index, columns=['Coord 1', 'Coord 2'])

fig = figure()
ax = Axes3D(fig)
cvalue = coords.values
names = list(coords.index)

for i in range(len(cvalue)): #plot each point + it's index as text above
    ax.scatter(cvalue[i, 0], cvalue[i, 1], 0, color='b')

plt.show()

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from pylab import figure

for i in range(2, 10):

    Z = linkage(coords.values, 'average')
    coords['Names'] = fcluster(Z, i, criterion='maxclust')
    print(Methods.dunnindex(coords, dissim))

coords.to_csv('../reaperCSVs/cluster data/clustering.csv')
fig = figure()
ax = Axes3D(fig)
cvalue=coords.values
names=list(coords.index)

for i in range(len(cvalue)): #plot each point + it's index as text above
    ax.scatter(cvalue[i,0], cvalue[i,1], cvalue[i,2], color='b')

for i in range(0,len(cvalue),20):
    ax.text(cvalue[i, 0], cvalue[i, 1], cvalue[i, 2], '%s' % (names[i]), size=5, zorder=1,
            color='k')
plt.show()

'4-4-1-66668-5-2018-08-01T18-15-25-1289'
'4-4-1-66668-4-2018-07-21T15-53-18-1026'
'4-2-1-62848-5-2018-03-22T07-17-39-904'
'4-3-2-65384-4-2018-06-12T15-25-28-1032'

#ax = plt.axes(projection='3d')
#ax.scatter3D(coords[:, 0], coords[:, 1], coords[:, 2], 'kx')
#plt.show()'''



'''removecols = ['Unnamed: 0', '0Frame_id', '0P1_mmr', '0P2_mmr',
              '0Replay_id', 'P1_Supply_army', 'P1_Supply_total', 'P1_Supply_used', 'P1_Supply_workers',
              'P1_minerals', 'P1_vespene', 'P2_Supply_army', 'P2_Supply_total', 'P2_Supply_used', 'P2_Supply_workers',
              'P2_minerals', 'P2_vespene']'''

'''removecols = ['Unnamed: 0', '0Frame_id', '0P1_mmr', '0P2_mmr',
              '0Replay_id', 'P1_Supply_army', 'P1_Supply_total', 'P1_Supply_used', 'P1_Supply_workers',
              'P1_minerals', 'P1_vespene',  'P2_Supply_army', 'P2_Supply_total', 'P2_Supply_used',
              'P2_Supply_workers', 'P2_Unit_Adept', 'P2_Unit_AdeptPhaseShift', 'P2_Unit_Archon',
              'P2_Unit_Assimilator', 'P2_Unit_Carrier', 'P2_Unit_Colossus', 'P2_Unit_CyberneticsCore',
              'P2_Unit_DarkShrine', 'P2_Unit_DarkTemplar', 'P2_Unit_Disruptor', 'P2_Unit_DisruptorPhased',
              'P2_Unit_FleetBeacon', 'P2_Unit_Forge', 'P2_Unit_Gateway', 'P2_Unit_HighTemplar', 'P2_Unit_Immortal',
              'P2_Unit_Interceptor', 'P2_Unit_Mothership', 'P2_Unit_Nexus', 'P2_Unit_Observer', 'P2_Unit_ObserverSurveillanceMode',
              'P2_Unit_Oracle', 'P2_Unit_Phoenix', 'P2_Unit_PhotonCannon', 'P2_Unit_Probe', 'P2_Unit_Pylon', 'P2_Unit_RoboticsBay',
              'P2_Unit_RoboticsFacility', 'P2_Unit_Sentry', 'P2_Unit_ShieldBattery', 'P2_Unit_Stalker', 'P2_Unit_Stargate',
              'P2_Unit_StasisTrap', 'P2_Unit_Tempest', 'P2_Unit_TemplarArchive', 'P2_Unit_TwilightCouncil', 'P2_Unit_VoidRay',
              'P2_Unit_WarpGate', 'P2_Unit_WarpPrism', 'P2_Unit_WarpPrismPhasing', 'P2_Unit_Zealot', 'P2_Upgrade_AdeptPiercingAttack',
              'P2_Upgrade_BlinkTech', 'P2_Upgrade_CARRIERLAUNCHSPEEDUPGRADE', 'P2_Upgrade_Charge', 'P2_Upgrade_DarkTemplarBlinkUpgrade',
              'P2_Upgrade_ExtendedThermalLance', 'P2_Upgrade_GraviticDrive', 'P2_Upgrade_ObserverGraviticBooster',
              'P2_Upgrade_PhoenixRangeUpgrade', 'P2_Upgrade_ProtossAirArmorsLevel1', 'P2_Upgrade_ProtossAirArmorsLevel2',
              'P2_Upgrade_ProtossAirArmorsLevel3', 'P2_Upgrade_ProtossAirWeaponsLevel1', 'P2_Upgrade_ProtossAirWeaponsLevel2',
              'P2_Upgrade_ProtossAirWeaponsLevel3', 'P2_Upgrade_ProtossGroundArmorsLevel1', 'P2_Upgrade_ProtossGroundArmorsLevel2',
              'P2_Upgrade_ProtossGroundArmorsLevel3', 'P2_Upgrade_ProtossGroundWeaponsLevel1', 'P2_Upgrade_ProtossGroundWeaponsLevel2',
              'P2_Upgrade_ProtossGroundWeaponsLevel3', 'P2_Upgrade_ProtossShieldsLevel1', 'P2_Upgrade_ProtossShieldsLevel2',
              'P2_Upgrade_ProtossShieldsLevel3', 'P2_Upgrade_PsiStormTech', 'P2_Upgrade_WarpGateResearch', 'P2_minerals', 'P2_vespene']'''


'''df = pd.read_csv('../reaperCSVs/cluster data/newsissimto360.0s.csv')
df.index = df['0Replay_id']
del df['0Replay_id']
print(df)
df = df.head(1000)
select = list(df.index)
df = df[select]
df = pd.DataFrame(df.values)
print(a.shape)
print(df.shape)'''

t = 360

dissim = pd.read_csv('../reaperCSVs/cluster data/newsissimto'+str(t)+'.0s.csv')
dissim.index = dissim['0Replay_id']
del dissim['0Replay_id']
#print(dissim)

samp = sample(list(dissim.index), 1000)
dissimtmp = dissim.loc[dissim.index.isin(samp)]
dissimtmp = dissimtmp.drop(list(set(dissimtmp.columns)-set(dissimtmp.index)), axis=1)
print(dissimtmp)

mds = manifold.MDS(n_components=3, dissimilarity="precomputed", random_state=6)
results = mds.fit(dissimtmp.values)
coords = results.embedding_

ax = plt.axes(projection='3d')
ax.scatter3D(coords[:, 0], coords[:, 1], coords[:, 2], 'kx')
plt.show()

'''silhouette = pd.DataFrame(index=range(2, 11))

for t in range(30, 870, 30):

    df = pd.read_csv('../reaperCSVs/cluster data/newcluster_data'+str(t)+'.0s.csv')
    df.index = df['0Replay_id']

    removecols = ['Unnamed: 0', 'Unnamed: 0.1', '0Frame_id', '0P1_mmr', '0P2_mmr',
                  '0Replay_id', 'P1_Supply_army', 'P1_Supply_total', 'P1_Supply_used', 'P1_Supply_workers',
                  'P1_minerals', 'P1_vespene', 'P2_Supply_army', 'P2_Supply_total', 'P2_Supply_used', 'P2_Supply_workers',
                  'P2_minerals', 'P2_vespene']

    df = df.drop(removecols, axis=1)
    df = Methods.removedoubles(df)

    dissim = pd.read_csv('../reaperCSVs/cluster data/distancematrixto'+str(t)+'.0s.csv')
    dissim.index = dissim['0Replay_id']
    del dissim['0Replay_id']
    df = df.drop([m for m in df.index if m not in dissim.index], axis=0)
    Z = linkage(ssd.squareform(dissim.values), method='single')

    print('t = '+str(t))
    sil = list()

    for i in range(2, 11):

        cls = fcluster(Z, i, criterion='maxclust')
        score = metrics.silhouette_score(dissim.values, cls, metric='precomputed')
        sil.append(score)
        print('Nr. of clusters = '+str(i)+': '+str(score))

    silhouette[t/60] = sil

silhouette.to_csv('../reaperCSVs/cluster data/silhouettescoressinglink.csv')'''

'''df = pd.read_csv('../reaperCSVs/cluster data/clusteringavg.csv')
df.index = df['Unnamed: 0']
del df['Unnamed: 0']
cls = list(df.index)
times = list()
cls2 = list()
mindists = list()

for c in df.columns:

    mind = list(df[c])
    mind = [m for m in mind if not np.isnan(m)]

    print('t = ' + str(float(c)/60)+' min.: ' + str(cls[mind.index(np.min(mind))]))
    times.append(float(c)/60)
    cls2.append(cls[mind.index(np.min(mind))])
    mindists.append(np.min(mind))

fig, ax = plt.subplots(nrows=2, ncols=1)
plt.subplot(2, 1, 1)
plt.plot(times, cls2)

plt.subplot(2, 1, 2)
plt.plot(times, mindists)
plt.show()

optimal = pd.DataFrame(columns=['Nr. of clusters', 'Dunn index'], index=times)
optimal['Nr. of clusters'] = cls2
optimal['Dunn index'] = mindists
optimal.to_csv('../reaperCSVs/cluster data/optimalclusterings.csv')'''

'''df720 = pd.read_csv('../reaperCSVs/cluster data/newcluster_data16560.csv')
print(list(df720.columns))
df720.index = df720['0Replay_id']
removecols = ['Unnamed: 0', 'Unnamed: 0.1', '0Frame_id', '0P1_mmr', '0P2_mmr',
              '0Replay_id', 'P1_Supply_army', 'P1_Supply_total', 'P1_Supply_used', 'P1_Supply_workers',
              'P1_minerals', 'P1_vespene', 'P2_Supply_army', 'P2_Supply_total', 'P2_Supply_used', 'P2_Supply_workers',
              'P2_minerals', 'P2_vespene']
              
df720 = df720.drop(removecols, axis=1)
print(df720)

clusteringavg = pd.DataFrame(index=[n for n in range(2, 11)])
clusteringminmax = pd.DataFrame(index=[n for n in range(2, 11)])

for t in range(720, 20880+720, 720):

    df720 = pd.read_csv('../reaperCSVs/cluster data/newcluster_data'+str(t)+'.csv')
    dissimilarity = pd.read_csv('../reaperCSVs/cluster data/distancematrixto'+str(t)+'.csv')
    dissimilarity.index=dissimilarity['0Replay_id']
    del dissimilarity['0Replay_id']

    expert = 0.1

    mindistsavg = list()
    mindistsminmax = list()

    cls = list()

    Z = linkage(ssd.squareform(dissimilarity.values), method='ward')

    for i in range(2, int(1 / expert) + 1):

        cl = fcluster(Z, i, criterion='maxclust')
        df720['Names'] = cl

        if all(i >= expert * len(df720.index) for i in [len(df720.loc[df720['Names'] == c].index) for c in list(set(cl))]):

            mindistsavg.append(Methods.dunnindexavg(df720, dissimilarity))
            mindistsminmax.append(Methods.dunnindex(df720, dissimilarity))
            cls.append(i)

    for c in cls:
        clusteringminmax.set_value(c, t, mindistsminmax[cls.index(c)])
        clusteringavg.set_value(c, t, mindistsavg[cls.index(c)])

    clusteringminmax.to_csv('../reaperCSVs/cluster data/10clusteringminmax.csv')
    clusteringavg.to_csv('../reaperCSVs/cluster data/10clusteringavg.csv')

    print(clusteringminmax)
    print(clusteringavg)'''


'''clusteringavg = pd.DataFrame(index=[n for n in range(2, 21)])
clusteringminmax = pd.DataFrame(index=[n for n in range(2, 21)])

removecols = ['Unnamed: 0', 'Unnamed: 0.1', '0Frame_id', '0P1_mmr', '0P2_mmr',
              '0Replay_id', 'P1_Supply_army', 'P1_Supply_total', 'P1_Supply_used', 'P1_Supply_workers',
              'P1_minerals', 'P1_vespene', 'P2_Supply_army', 'P2_Supply_total', 'P2_Supply_used', 'P2_Supply_workers',
              'P2_minerals', 'P2_vespene']

df720 = pd.read_csv('../reaperCSVs/cluster data/newcluster_data720.csv')
df720.index = df720['0Replay_id']
df720 = Methods.removedoubles(df720)

df720 = df720.drop(removecols, axis=1)

matches = list(df720.index)

dissimilarity = pd.DataFrame(distance_matrix(df720.values, df720.values, p=1),
                                        columns=df720.index, index=df720.index)
dissimilarity.to_csv('../reaperCSVs/cluster data/distancematrixto720.csv', encoding='utf-8', index=True)

expert = 0.05

mindistsavg = list()
mindistsminmax = list()
cls = list()

Z = linkage(ssd.squareform(dissimilarity.values), method='ward')

for i in range(2, int(1 / expert) + 1):

    cl = fcluster(Z, i, criterion='maxclust')
    df720['Names'] = cl

    if all(i >= expert * len(df720.index) for i in [len(df720.loc[df720['Names'] == c].index) for c in list(set(cl))]):

        mindistsavg.append(Methods.dunnindexavg(df720, dissimilarity))
        mindistsminmax.append(Methods.dunnindex(df720, dissimilarity))
        cls.append(i)

for c in cls:

    clusteringminmax.set_value(c, 720, mindistsminmax[cls.index(c)])
    clusteringavg.set_value(c, 720, mindistsavg[cls.index(c)])

clusteringminmax.to_csv('../reaperCSVs/cluster data/clusteringminmax.csv')
clusteringavg.to_csv('../reaperCSVs/cluster data/clusteringavg.csv')

print(clusteringminmax)
print(clusteringavg)

for t in range(720+720, 20880+720, 720):

    dftmp = pd.read_csv('../reaperCSVs/cluster data/newcluster_data'+str(t)+'.csv')

    dftmp.index = dftmp['0Replay_id']
    dftmp = Methods.removedoubles(dftmp)
    dftmp = dftmp.drop(removecols, axis=1)

    matches = [m for m in matches if m in list(dftmp.index)]

    dftmp = dftmp.loc[matches]
    dissimilarity = dissimilarity[matches].loc[matches]

    dissimilarity = pd.DataFrame(data=distance_matrix(dftmp.values, dftmp.values, p=1) + dissimilarity.values,
                                     columns=dissimilarity.columns, index=dissimilarity.index)

    dissimilarity.to_csv('../reaperCSVs/cluster data/distancematrixto'+str(t)+'.csv', encoding='utf-8', index=True)

    mindistsavg = list()
    mindistsminmax = list()
    cls = list()

    Z = linkage(ssd.squareform(dissimilarity.values), method='ward')

    for i in range(2, int(1 / expert) + 1):

        cl = fcluster(Z, i, criterion='maxclust')
        dftmp['Names'] = cl

        if all(i >= expert * len(dftmp.index) for i in [len(dftmp.loc[dftmp['Names'] == c].index) for c in list(set(cl))]):

            mindistsavg.append(Methods.dunnindexavg(dftmp, dissimilarity))
            mindistsminmax.append(Methods.dunnindex(dftmp, dissimilarity))
            cls.append(i)

    for c in cls:

        clusteringminmax.set_value(c, t, mindistsminmax[cls.index(c)])
        clusteringavg.set_value(c, t, mindistsavg[cls.index(c)])

    clusteringminmax.to_csv('../reaperCSVs/cluster data/clusteringminmax.csv')
    clusteringavg.to_csv('../reaperCSVs/cluster data/clusteringavg.csv')

    print(clusteringavg)
    print(clusteringminmax)'''

'''df = pd.read_csv('../reaperCSVs/cluster data/cluster_data20880.csv')
colmax = df.columns

for t in range(720, 20880+720, 720):

    dftmp = pd.read_csv('../reaperCSVs/cluster data/cluster_data' + str(t) + '.csv')
    df = pd.DataFrame(data=np.zeros([len(dftmp.index), len(colmax)]), columns=colmax, index=dftmp.index)

    for c in dftmp.columns:
        df[c] = dftmp[c]

    df.to_csv('../reaperCSVs/cluster data/newcluster_data'+str(t)+'.csv')


optis = pd.DataFrame(index=['Nr. of clusters', 'Dunn Index'])

for t in range(60, 600+30, 30):

    cls, mindists = Methods.checkOptimalClustering(t, 0.05)

    optis[t] = [int(cls[mindists.index(np.min(mindists))]), np.min(mindists)]

    print(optis)

optis.to_csv('../data/optimalclustering.csv')'''

removecols = ['Unnamed: 0', 'Unnamed: 0.1', '0Frame_id', '0P1_mmr', '0P2_mmr',
              '0Replay_id', 'P1_Supply_army', 'P1_Supply_total', 'P1_Supply_used', 'P1_Supply_workers',
              'P1_minerals', 'P1_vespene',  'P2_Supply_army', 'P2_Supply_total', 'P2_Supply_used',
              'P2_Supply_workers', 'P2_Unit_Adept', 'P2_Unit_AdeptPhaseShift', 'P2_Unit_Archon',
              'P2_Unit_Assimilator', 'P2_Unit_Carrier', 'P2_Unit_Colossus', 'P2_Unit_CyberneticsCore',
              'P2_Unit_DarkShrine', 'P2_Unit_DarkTemplar', 'P2_Unit_Disruptor', 'P2_Unit_DisruptorPhased',
              'P2_Unit_FleetBeacon', 'P2_Unit_Forge', 'P2_Unit_Gateway', 'P2_Unit_HighTemplar', 'P2_Unit_Immortal',
              'P2_Unit_Interceptor', 'P2_Unit_Mothership', 'P2_Unit_Nexus', 'P2_Unit_Observer', 'P2_Unit_ObserverSurveillanceMode',
              'P2_Unit_Oracle', 'P2_Unit_Phoenix', 'P2_Unit_PhotonCannon', 'P2_Unit_Probe', 'P2_Unit_Pylon', 'P2_Unit_RoboticsBay',
              'P2_Unit_RoboticsFacility', 'P2_Unit_Sentry', 'P2_Unit_ShieldBattery', 'P2_Unit_Stalker', 'P2_Unit_Stargate',
              'P2_Unit_StasisTrap', 'P2_Unit_Tempest', 'P2_Unit_TemplarArchive', 'P2_Unit_TwilightCouncil', 'P2_Unit_VoidRay',
              'P2_Unit_WarpGate', 'P2_Unit_WarpPrism', 'P2_Unit_WarpPrismPhasing', 'P2_Unit_Zealot', 'P2_Upgrade_AdeptPiercingAttack',
              'P2_Upgrade_BlinkTech', 'P2_Upgrade_CARRIERLAUNCHSPEEDUPGRADE', 'P2_Upgrade_Charge', 'P2_Upgrade_DarkTemplarBlinkUpgrade',
              'P2_Upgrade_ExtendedThermalLance', 'P2_Upgrade_GraviticDrive', 'P2_Upgrade_ObserverGraviticBooster',
              'P2_Upgrade_PhoenixRangeUpgrade', 'P2_Upgrade_ProtossAirArmorsLevel1', 'P2_Upgrade_ProtossAirArmorsLevel2',
              'P2_Upgrade_ProtossAirArmorsLevel3', 'P2_Upgrade_ProtossAirWeaponsLevel1', 'P2_Upgrade_ProtossAirWeaponsLevel2',
              'P2_Upgrade_ProtossAirWeaponsLevel3', 'P2_Upgrade_ProtossGroundArmorsLevel1', 'P2_Upgrade_ProtossGroundArmorsLevel2',
              'P2_Upgrade_ProtossGroundArmorsLevel3', 'P2_Upgrade_ProtossGroundWeaponsLevel1', 'P2_Upgrade_ProtossGroundWeaponsLevel2',
              'P2_Upgrade_ProtossGroundWeaponsLevel3', 'P2_Upgrade_ProtossShieldsLevel1', 'P2_Upgrade_ProtossShieldsLevel2',
              'P2_Upgrade_ProtossShieldsLevel3', 'P2_Upgrade_PsiStormTech', 'P2_Upgrade_WarpGateResearch', 'P2_minerals', 'P2_vespene']

'''from sklearn.preprocessing import StandardScaler

standard = StandardScaler()

df30 = pd.read_csv('../reaperCSVs/cluster data/newcluster_data30.0s.csv')
df30.index = df30['0Replay_id']
del df30['0Replay_id']
df30 = Methods.removedoubles(df30)
matches = list(df30.index)

dissim = pd.DataFrame(distance_matrix(standard.fit_transform(df30.values), standard.fit_transform(df30.values), p=1),
                      index=df30.index, columns=df30.index)
dissim.to_csv('../reaperCSVs/cluster data/newsissimto30.0s.csv',float_format='%.1f')

for t in range(60, 840+30, 30):

    dftmp = pd.read_csv('../reaperCSVs/cluster data/newcluster_data'+str(t)+'.0s.csv')
    dftmp.index = dftmp['0Replay_id']
    del dftmp['0Replay_id']
    dftmp = Methods.removedoubles(dftmp)

    matches = [m for m in matches if m in list(dftmp.index)]

    dftmp = dftmp.loc[dftmp.index.isin(matches)]
    dissim = dissim.drop(list(set(dissim.columns) - set(matches)), axis=1)
    dissim = dissim.loc[dissim.index.isin(matches)]

    values = distance_matrix(standard.fit_transform(dftmp.values), standard.fit_transform(dftmp.values), p=1) + dissim.values

    dissim = pd.DataFrame(data=values, columns=dissim.index, index=dissim.index)
    dissim.to_csv('../reaperCSVs/cluster data/newsissimto'+str(t)+'.0s.csv', float_format='%.1f')

    print('t='+str(t)+' completed', end='\r')'''



'''df30 = pd.read_csv('../BuildOrderGameCSVsV2/buildordersto30s.csv')
df30.index = df30['Unnamed: 0']
del df30['Unnamed: 0']
df30 = Methods.removedoubles(df30)

dissimilarity = pd.DataFrame(data=distance_matrix(df30.values, df30.values, p=1), columns=df30.index, index=df30.index)
dissimilarity.to_csv('../BuildOrderGameCSVsV2/2dissimilaritymatrixto30.csv', encoding='utf-8', index=True)

for t in range(60, 600+30, 30):

    matches = list(dissimilarity.index)

    dftmp = pd.read_csv('../BuildOrderGameCSVsV2/buildordersto'+str(t)+'s.csv')
    dftmp.index = dftmp['Unnamed: 0']
    del dftmp['Unnamed: 0']
    dftmp = Methods.removedoubles(dftmp)
    matches2 = list(dftmp.index)

    for m in matches:

        if not (m in matches2):

            dissimilarity = dissimilarity.drop(m, axis=0)
            dissimilarity = dissimilarity.drop(m, axis=1)
            matches.remove(m)

    dftmp = dftmp[dftmp.index.isin(matches)]

    for m in np.setdiff1d(dissimilarity.index, dftmp.index):
        dissimilarity = dissimilarity.drop(m, axis=0)
        dissimilarity = dissimilarity.drop(m, axis=1)
        matches.remove(m)

    values = distance_matrix(dftmp.values, dftmp.values, p=1)

    values = values + dissimilarity.values
    dissimilarity = pd.DataFrame(data=values, columns=dissimilarity.index, index=dissimilarity.index)
    #print(dissimilarity)
    dissimilarity.to_csv('../BuildOrderGameCSVsV2/2dissimilaritymatrixto'+str(t)+'.csv',
                         encoding='utf-8', index=True)
    #print(dissimilarity.shape)
    print('t='+str(t)+' completed', end='\r')'''

'''builds = ['ArchonImmortal', 'BlinkStalkers', 'DarkTemplarRush',
          'OracleOpening', 'PheonixOpening', 'ZealotArchon']

cols = ['Adept', 'Archon', 'Carrier', 'Colossus', 'DarkTemplar', 'Disruptor',
       'HighTemplar', 'Immortal', 'Mothership', 'Observer', 'Oracle',
       'Phoenix', 'Probe', 'Sentry', 'Stalker', 'Tempest', 'WarpPrism',
       'VoidRay', 'Zealot', 'Assimilator', 'CyberneticsCore', 'DarkShrine',
       'FleetBeacon', 'Forge', 'Gateway', 'Nexus', 'PhotonCannon', 'Pylon',
       'RoboticsFacility', 'RoboticsBay', 'ShieldBattery', 'Stargate',
       'TemplarArchive', 'TwilightCouncil', 'WarpGate',
       'ProtossAirArmorsLevel1', 'ProtossAirArmorsLevel2',
       'ProtossAirArmorsLevel3', 'ProtossAirWeaponsLevel1',
       'ProtossAirWeaponsLevel2', 'ProtossAirWeaponsLevel3',
       'ProtossGroundArmorsLevel1', 'ProtossGroundArmorsLevel2',
       'ProtossGroundArmorsLevel3', 'ProtossGroundWeaponsLevel1',
       'ProtossGroundWeaponsLevel2', 'ProtossGroundWeaponsLevel3',
       'ProtossShieldsLevel1', 'ProtossShieldsLevel2', 'ProtossShieldsLevel3',
       'AdeptPiercingAttack', 'BlinkTech', 'Charge', 'DarkTemplarBlinkUpgrade',
       'ExtendedThermalLance', 'GraviticDrive', 'ObserverGraviticBooster',
       'PhoenixRangeUpgrade', 'PsiStormTech', 'WarpGateResearch']

cols2 = ['Adept1', 'Archon1', 'Carrier1', 'Colossus1', 'DarkTemplar1', 'Disruptor1',
       'HighTemplar1', 'Immortal1', 'Mothership1', 'Observer1', 'Oracle1',
       'Phoenix1', 'Probe1', 'Sentry1', 'Stalker1', 'Tempest1', 'WarpPrism1',
       'VoidRay1', 'Zealot1', 'Assimilator1', 'CyberneticsCore1',
       'DarkShrine1', 'FleetBeacon1', 'Forge1', 'Gateway1', 'Nexus1',
       'PhotonCannon1', 'Pylon1', 'RoboticsFacility1', 'RoboticsBay1',
       'ShieldBattery1', 'Stargate1', 'TemplarArchive1', 'TwilightCouncil1',
       'WarpGate1', 'ProtossAirArmorsLevel11', 'ProtossAirArmorsLevel21',
       'ProtossAirArmorsLevel31', 'ProtossAirWeaponsLevel11',
       'ProtossAirWeaponsLevel21', 'ProtossAirWeaponsLevel31',
       'ProtossGroundArmorsLevel11', 'ProtossGroundArmorsLevel21',
       'ProtossGroundArmorsLevel31', 'ProtossGroundWeaponsLevel11',
       'ProtossGroundWeaponsLevel21', 'ProtossGroundWeaponsLevel31',
       'ProtossShieldsLevel11', 'ProtossShieldsLevel21',
       'ProtossShieldsLevel31', 'AdeptPiercingAttack1', 'BlinkTech1',
       'Charge1', 'DarkTemplarBlinkUpgrade1', 'ExtendedThermalLance1',
       'GraviticDrive1', 'ObserverGraviticBooster1', 'PhoenixRangeUpgrade1',
       'PsiStormTech1', 'WarpGateResearch1']

for t in range(30, 630, 30):

    buildorders = pd.read_csv('../BuildOrderGameCSVsV2/1GateExpandCSV-'+str(t)+'s.csv')
    del buildorders['Unnamed: 0']
    buildorders.index = buildorders['Name']
    del buildorders['Name']
    dftmp = buildorders[cols2]

    for c in cols2:
        del buildorders[c]

    newindex=[i + 'b' for i in buildorders.index]

    buildorders = buildorders.append(pd.DataFrame(dftmp.values, columns=cols, index=newindex))

    for b in builds:

        df = pd.read_csv('../BuildOrderGameCSVsV2/'+b+'CSV-'+str(t)+'s.csv')
        del df['Unnamed: 0']
        df.index = df['Name']
        del df['Name']
        newindex2 = [i + 'b' for i in df.index]

        buildorders = buildorders.append(pd.DataFrame(df[cols2].values, columns=cols2,index=newindex2))

    buildorders.to_csv('../BuildOrderGameCSVsV2/buildordersto'+str(t)+'s.csv')
    print('t='+str(t)+'         '+str(buildorders.shape))
'''


'''df30 = pd.read_csv('../BuildOrderCSVs/buildordersto90s.csv')
del df30['Unnamed: 0']
df30=df30.dropna()

dissimilarity = distance_matrix(df60.values, df60.values, p=1)

for t in range(90, 210 + 30, 30):

    dftmp = pd.read_csv('../BuildOrderCSVs/buildordersto'+str(t)+'s.csv')
    del dftmp['Unnamed: 0']
    dftmp=dftmp.dropna()
    print('t='+str(t)+'    '+str(dftmp.shape))

    values = distance_matrix(dftmp.values, dftmp.values, p=1) + dissimilarity
    dissimilarity = pd.DataFrame(values)

    dissimilarity.to_csv('../BuildOrderCSVs/bodissimilaritymatrixto'+str(t)+'.csv', encoding='utf-8', index=True)
    dissimilarity = dissimilarity.values
    print(dissimilarity)
    #print('t=' + str(t) + ' completed', end='\r')'''

'''for t in range(30, 600+30, 30):

    dissim = pd.read_csv('../BuildOrderGameCSVsV2/2dissimilaritymatrixto' + str(t) + '.csv')

    dissim.index = dissim['Unnamed: 0']
    del dissim['Unnamed: 0']

    df = pd.read_csv('../BuildOrderGameCSVsV2/buildordersto' + str(t) + 's.csv')
    df.index = df['Unnamed: 0']
    del df['Unnamed: 0']
    df = df.loc[df.index.isin(dissim.index)]

    mind = list()
    cls = list()

    distArray = ssd.squareform(dissim.values)
    Z = linkage(distArray, method='ward')

    for i in range(2, int(1/0.05)+1):

        cl = fcluster(Z, i, criterion='maxclust')
        df['Names'] = cl
        size = len(df.index)

        check = True

        for c in list(set(cl)):

            dftmp = df.loc[df['Names'] == c]
            clsize = len(dftmp.index)

            if clsize < 0.05*size:
                check = False

        if check:

            mind.append(Methods.dunnindex(df, dissim))
            cls.append(i)

        #print('t='+str(t)+': '+str(i)+'/'+str(int(1/0.05))+' clusters checked.', end='\r')

    print(cls)
    print(mind)'''






'''for t in range(60, 210 + 30, 30):

    dftmp = pd.read_csv('../BuildOrderCSVs/buildordersto'+str(t)+'s.csv')
    del dftmp['Unnamed: 0']
    print(t)
    print(dftmp)
    dftmp.dropna()
    print(dftmp)

df60 = pd.read_csv('../BuildOrderCSVs/buildordersto90s.csv')
del df60['Unnamed: 0']
df60=df60.dropna()

dissimilarity = distance_matrix(df60.values, df60.values, p=1)

for t in range(90, 210 + 30, 30):

    dftmp = pd.read_csv('../BuildOrderCSVs/buildordersto'+str(t)+'s.csv')
    del dftmp['Unnamed: 0']
    dftmp=dftmp.dropna()
    print('t='+str(t)+'    '+str(dftmp.shape))

    values = distance_matrix(dftmp.values, dftmp.values, p=1) + dissimilarity
    dissimilarity = pd.DataFrame(values)

    dissimilarity.to_csv('../BuildOrderCSVs/bodissimilaritymatrixto'+str(t)+'.csv', encoding='utf-8', index=True)
    dissimilarity = dissimilarity.values
    print(dissimilarity)
    #print('t=' + str(t) + ' completed', end='\r')'''





'''builds = ['ArchonImmortal', 'BlinkStalkers', 'DarkTemplarRush',
          'OracleOpening', 'PheonixOpening', 'ZealotArchon']

cols = ['Adept', 'Archon', 'Carrier', 'Colossus', 'DarkTemplar', 'Disruptor',
       'HighTemplar', 'Immortal', 'Mothership', 'Observer', 'Oracle',
       'Phoenix', 'Probe', 'Sentry', 'Stalker', 'Tempest', 'WarpPrism',
       'VoidRay', 'Zealot', 'Assimilator', 'CyberneticsCore', 'DarkShrine',
       'FleetBeacon', 'Forge', 'Gateway', 'Nexus', 'PhotonCannon', 'Pylon',
       'RoboticsFacility', 'RoboticsBay', 'ShieldBattery', 'Stargate',
       'TemplarArchive', 'TwilightCouncil', 'WarpGate',
       'ProtossAirArmorsLevel1', 'ProtossAirArmorsLevel2',
       'ProtossAirArmorsLevel3', 'ProtossAirWeaponsLevel1',
       'ProtossAirWeaponsLevel2', 'ProtossAirWeaponsLevel3',
       'ProtossGroundArmorsLevel1', 'ProtossGroundArmorsLevel2',
       'ProtossGroundArmorsLevel3', 'ProtossGroundWeaponsLevel1',
       'ProtossGroundWeaponsLevel2', 'ProtossGroundWeaponsLevel3',
       'ProtossShieldsLevel1', 'ProtossShieldsLevel2', 'ProtossShieldsLevel3',
       'AdeptPiercingAttack', 'BlinkTech', 'Charge', 'DarkTemplarBlinkUpgrade',
       'ExtendedThermalLance', 'GraviticDrive', 'ObserverGraviticBooster',
       'PhoenixRangeUpgrade', 'PsiStormTech', 'WarpGateResearch']

cols2 = ['Adept1', 'Archon1', 'Carrier1', 'Colossus1', 'DarkTemplar1', 'Disruptor1',
       'HighTemplar1', 'Immortal1', 'Mothership1', 'Observer1', 'Oracle1',
       'Phoenix1', 'Probe1', 'Sentry1', 'Stalker1', 'Tempest1', 'WarpPrism1',
       'VoidRay1', 'Zealot1', 'Assimilator1', 'CyberneticsCore1',
       'DarkShrine1', 'FleetBeacon1', 'Forge1', 'Gateway1', 'Nexus1',
       'PhotonCannon1', 'Pylon1', 'RoboticsFacility1', 'RoboticsBay1',
       'ShieldBattery1', 'Stargate1', 'TemplarArchive1', 'TwilightCouncil1',
       'WarpGate1', 'ProtossAirArmorsLevel11', 'ProtossAirArmorsLevel21',
       'ProtossAirArmorsLevel31', 'ProtossAirWeaponsLevel11',
       'ProtossAirWeaponsLevel21', 'ProtossAirWeaponsLevel31',
       'ProtossGroundArmorsLevel11', 'ProtossGroundArmorsLevel21',
       'ProtossGroundArmorsLevel31', 'ProtossGroundWeaponsLevel11',
       'ProtossGroundWeaponsLevel21', 'ProtossGroundWeaponsLevel31',
       'ProtossShieldsLevel11', 'ProtossShieldsLevel21',
       'ProtossShieldsLevel31', 'AdeptPiercingAttack1', 'BlinkTech1',
       'Charge1', 'DarkTemplarBlinkUpgrade1', 'ExtendedThermalLance1',
       'GraviticDrive1', 'ObserverGraviticBooster1', 'PhoenixRangeUpgrade1',
       'PsiStormTech1', 'WarpGateResearch1']

for t in range(60, 630, 30):

    buildorders = pd.read_csv('../BuildOrderCSVs/1GateExpandCSV-'+str(t)+'s.csv')
    del buildorders['Unnamed: 0']
    dftmp = buildorders[cols2]

    for c in cols2:
        del buildorders[c]

    buildorders = buildorders.append(pd.DataFrame(dftmp.values, columns=cols))

    for b in builds:

        df = pd.read_csv('../BuildOrderCSVs/'+b+'CSV-'+str(t)+'s.csv')
        del df['Unnamed: 0']

        buildorders = buildorders.append(pd.DataFrame(df[cols2].values, columns=cols))

    buildorders.index = [q for q in range(1, len(buildorders.index)+1)]
    buildorders.to_csv('../BuildOrderCSVs/buildordersto'+str(t)+'s.csv')
    print('t='+str(t)+'         '+str(buildorders.shape))'''

'''optimalcls = list()
dunnindexes = list()
times=list()

for t in range(90, 630, 30):

    cl, dunn = Methods.checkOptimalClustering(t, 0.1)

    if not (cl == []):
        optimalcls.append(cl[dunn.index(np.min(dunn))])
        dunnindexes.append(np.min(dunn))
        times.append(t)
'''

optimalcls = [2, 2, 2,
              2, 3, 2,
              2, 2, 3,
              2, 3, 3,
              3, 4, 6]


dunnindexes = [84.43893116636282,  45.24025063504096,  14.466880089388129,
               10.647518630469945, 7.427834615992336,  4.230495869587379,
               2.9149050396911473, 2.374869001757376,  1.4302009845468617,
               1.551624348313532,  1.265030281225694,  1.0177125638557059,
               0.9029037692623603, 0.7777417432318938, 0.4052641418131906]


times = [90,  120, 210,
         240, 270, 330,
         360, 390, 420,
         450, 480, 510,
         540, 570, 600]


