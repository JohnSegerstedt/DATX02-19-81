from clustering.Utilities import Methods
from clustering.Utilities import Classes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
import seaborn as sns

#A = Methods.initTmp()


data = genfromtxt(r'C:\Users\arvid\Desktop\Skola\Skolår 3\Kandidatarbete\MagicTelescope.csv', delimiter=',')
#data = data[1:5000,1:11]
data = data[2000:5000,:]
df = (pd.DataFrame(data)-pd.DataFrame(data).mean())/pd.DataFrame(data).std()


Methods.heatMap(df) # Undesök korrelation i data


arr = input("Columns?") #input vilka colomner som ska undersökas
cols = list(map(int,arr.split(' ')))
df = df[cols]


df_DBSCAN = Methods.cluster_DBSCAN2(df=df, eps=.3, min_samples=10, keepOutliers=True, keepVarnames=True)
Methods.project_onto_R3(df_DBSCAN, [0, 1, 2])
df_DBSCAN = df_DBSCAN.loc[df_DBSCAN['Names'] != 'Outlier']
Methods.parallelCoordinates(df_DBSCAN)

k = int(input('k in KMeans?'))
df_KMeans = Methods.cluster_KMeans2(df_DBSCAN,k)
Methods.project_onto_R3(df_KMeans, [0, 1, 2])


#Methods.wardLinkage(df)

#Methods.heatMap(data)



