from clustering.Utilities import Methods
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import SparsePCA,PCA

df = pd.read_csv(r'../data/Replays2-210.0s.csv')
df=df.drop('Unnamed: 0', axis=1)
print(df.shape)
df = df.loc[:, (df != 0).any(axis=0)]
print(df.shape)
Methods.explainedVariance(df)
df=Methods.clusterPCA(df,3)
print(df)
df_cluster=Methods.cluster_KMeans2(df,3,True,True)
Methods.project_onto_R32(df,['PCA 0','PCA 1','PCA 2'])

#Methods.heatMap(df)
#Methods.seabornHeatmap(df)
#Methods.explainedVariance(df)

#df_pca=Methods.clusterPCA(df,2)
#inverse=Methods.inversePCA(df,0.95)
#Methods.project_onto_R32(inverse,[0,1,2])
#df_new=Methods.cluster_KMeans2(inverse,2,True,True)
#print(df_new)
#Methods.project_onto_R32(df_new,[0,1,2])
