from clustering.Utilities import Methods
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import SparsePCA,PCA

df = pd.read_csv(r'../data/Replays2-210.0s.csv')
df=df.drop('Unnamed: 0', axis=1)

df = df.loc[:, (df != 0).any(axis=0)]
Methods.heatMap(df)
Methods.explainedVariance(df)
df=Methods.clusterPCA(df,3)

df_cluster=Methods.cluster_KMeans(df,3,True,True)
Methods.project_onto_R3(df,['PCA 0','PCA 1','PCA 2'])
