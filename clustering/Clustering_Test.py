from clustering.Utilities import Methods
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'../data/Replays2-210.0s.csv')
df=df.drop('Unnamed: 0', axis=1)
df = df.loc[:, (df != 0).any(axis=0)]

Methods.explainedVariance(df)

df_cluster=Methods.cluster_KMeans(Methods.clusterPCA(df,3),2,True)

df['Names']=df_cluster['Names']
df=Methods.binaryCluster(df)

corr=Methods.heatMap(df)
