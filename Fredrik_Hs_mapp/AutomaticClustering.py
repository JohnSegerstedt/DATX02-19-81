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
import os

distancemetric = 'Cosine'            #Vilket distansmått man ska använda
finalt = 2160                           #Hur sent in i matchen man vill kolla (i frames)
scale = True                            #Om datan ska skalas med MinMax vid omräkning av distansmatriserna
clusteringmethod = 'hierarchical'       #Vilken klustringsmetod
linkagetype = 'ward'                    #Vilken linkage type i hierarkisk klustirng
nrofclusters = 3                        #Hur många kluster man vill klustra på
PCA = False

removecols = ['0P1_result', '0P2_result', 'Unnamed: 0', '0Frame_id', '0P1_mmr',
              '0P2_mmr', '0Replay_id', 'P1_Supply_army', 'P1_Supply_total',
              'P1_Supply_used', 'P1_Supply_workers', 'P1_minerals', 'P1_vespene',
              'P2_Supply_army', 'P2_Supply_total', 'P2_Supply_used', 'P2_Supply_workers',
              'P2_minerals', 'P2_vespene']

if 'pictureto'+str(finalt)+'.csv' not in os.listdir('../Fredrik_Hs_mapp/PictureFiles/'):

    if 'verificationfileP2_Upgrade_WarpGateResearch' + str(finalt) + '.csv' not in os.listdir(
            '../Fredrik_Hs_mapp/VerificationFiles/'):

        df = pd.read_csv('../reaperCSVs/cluster data/cluster_data' + str(finalt) + '.csv')
        df = df.drop(removecols, axis=1)

        i = 0

        for feature in df.columns:

            Methods.verification(feature, finalt)
            i = i + 1
            print(str(i) + '/' + str(len(df.columns)), end='\r')

    df = pd.read_csv('../reaperCSVs/cluster data/cluster_data'+str(finalt)+'.csv')
    df = df.drop(removecols, axis=1)
    df2 = pd.read_csv('../Fredrik_Hs_mapp/VerificationsFiles/verificationfileP1_Unit_Probe'+str(finalt)+'.csv')
    matches = list(df2['Unnamed: 0'])

    picture = pd.DataFrame(index=matches)

    i = 0

    for cols in list(df.columns):

        dftmp = pd.read_csv('../Fredrik_Hs_mapp/VerificationFiles/verificationfile' + cols + str(finalt)+'.csv')
        dftmp.index = dftmp['Unnamed: 0']
        del dftmp['Unnamed: 0']

        for t in dftmp.columns:

            picture[cols+str(t)] = dftmp[t].values

        i = i+1

        #print(str(i)+'/'+str(len(df.columns)), end='\r')

    picture.to_csv('../Fredrik_Hs_mapp/PictureFiles/pictureto'+str(finalt)+'.csv')

df = pd.read_csv('../Fredrik_Hs_mapp/PictureFiles/pictureto'+str(finalt)+'.csv')
df.index = df['Unnamed: 0']
del df['Unnamed: 0']
df = df[df.index.notnull()]

if PCA:
    df = Methods.getPCs(df, 3)
    print(df)
    Methods.project_onto_R3(df, ['PC 1', 'PC 2', 'PC 3'])
    plt.show()

if scale:
    if 'scaled' + distancemetric + 'distancematrixto' + str(finalt) + '.csv' not in os.listdir('../Fredrik_Hs_mapp/DistansMatriser/'):

        dissim = pd.DataFrame(np.around(squareform(pdist(MinMaxScaler().fit_transform(df.values),
                                                         distancemetric)), 3), columns=df.index, index=df.index)
        dissim.to_csv(
            '../Fredrik_Hs_mapp/DistansMatriser/' + 'scaled' + distancemetric + 'distancematrixto' + str(finalt) + '.csv')
    else:
        dissim = pd.read_csv( '.../Fredrik_Hs_mapp/DistansMatriser/' + 'scaled' + distancemetric + 'distancematrixto' + str(finalt) + '.csv')
        dissim.index=dissim['Unnamed: 0']
        del dissim['Unnamed: 0']
    if 'MDS' + 'scaled'+distancemetric + 'distancematrixto' + str(finalt) + '.csv' not in os.listdir('../Fredrik_Hs_mapp/MDSfiler/'):
        mds = manifold.MDS(n_components=3, dissimilarity="precomputed", random_state=5, verbose=2)
        results = mds.fit(dissim.values)
        coords = pd.DataFrame(results.embedding_, index=dissim.index, columns=['Coord 1', 'Coord 2', 'Coord 3'])
        coords.to_csv('../Fredrik_Hs_mapp/MDSfiler/' + 'MDS' + 'scaled' + distancemetric + 'distancematrixto' + str(finalt) + '.csv')
    else:
        coords = pd.read_csv('../reaperCSVs/cluster data/'+'MDS' + 'scaled'+distancemetric + 'distancematrixto' + str(finalt) + '.csv')
        coords.index = coords['Unnamed: 0']
        del coords['Unnamed: 0']
else:
    if distancemetric + 'distancematrixto' + str(finalt) + '.csv' not in os.listdir('../Fredrik_Hs_mapp/DistansMatriser/'):
        dissim = pd.DataFrame(np.around(squareform(pdist(df.values, distancemetric)), 3), columns=df.index, index=df.index)
        dissim.to_csv('../Fredrik_Hs_mapp/DistansMatriser/' + distancemetric + 'distancematrixto' + str(finalt) + '.csv')
    else:
        dissim = pd.read_csv('../Fredrik_Hs_mapp/DistansMatriser/' + distancemetric + 'distancematrixto' + str(finalt) + '.csv')
        dissim.index = dissim['Unnamed: 0']
        del dissim['Unnamed: 0']
    if 'MDS'+distancemetric + 'distancematrixto' + str(finalt) + '.csv' not in os.listdir('../Fredrik_Hs_mapp/MDSfiler/'):
        mds = manifold.MDS(n_components=3, dissimilarity="precomputed", random_state=5, verbose=2)
        results = mds.fit(dissim.values)
        coords = pd.DataFrame(results.embedding_, index=dissim.index, columns=['Coord 1', 'Coord 2', 'Coord 3'])
        coords.to_csv('../Fredrik_Hs_mapp/MDSfiler/' + 'MDS' + distancemetric + 'distancematrixeto' + str(finalt) + '.csv')
    else:
        coords = pd.read_csv('../Fredrik_Hs_mapp/MDSfiler/'+'MDS' +distancemetric + 'distancematrixto' + str(finalt) + '.csv')
        coords.index = coords['Unnamed: 0']
        del coords['Unnamed: 0']

if clusteringmethod == 'spectral':
    sim = Methods.dissimtosim(dissim)
    clustering = SpectralClustering(n_clusters=nrofclusters, assign_labels="discretize", random_state=0, affinity='precomputed').fit(sim.values)
    coords['Names'] = clustering.labels_
elif clusteringmethod == 'hierarchical':
    Z = linkage(squareform(dissim.values), linkagetype)
    coords['Names'] = fcluster(Z, nrofclusters, criterion='maxclust')

clustering = pd.DataFrame(index=coords.index)
clustering['Cluster'] = coords['Names']

if clusteringmethod == 'spectral':
    clustering.to_csv('../reaperCSVs/cluster data/'+str(nrofclusters)+'clustering to'+str(finalt)+'using'+distancemetric+
                      'and'+clusteringmethod+'.csv')
elif clusteringmethod == 'hierarchical':
    clustering.to_csv('../reaperCSVs/cluster data/' + str(nrofclusters) + 'clustering to' + str(
        finalt) + 'using' + distancemetric + 'and' + clusteringmethod + 'with'+linkagetype+'linkage.csv')

Methods.project_onto_R3(coords, ['Coord 1', 'Coord 2', 'Coord 3'])
plt.show()

