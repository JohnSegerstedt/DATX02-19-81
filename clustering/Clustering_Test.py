from clustering.Utilities import Methods
import pandas as pd



df = pd.read_csv(r'../data/Replays2-210.0s.csv')
df=df.drop('Unnamed: 0', axis=1)
df = df.loc[:, (df != 0).any(axis=0)]


df=Methods.clusterPCA(df,3)

df_cluster=Methods.cluster_KMeans(df,2,True)
df_cluster=Methods.inversePCA(df_cluster,0.95)
df_DBSCAN = Methods.cluster_DBSCAN(df_cluster,0.4,10,False,True)
Methods.project_onto_R3(df_DBSCAN,[0,1,2])
cluster_2 = df_DBSCAN.loc[df_DBSCAN['Names'] == 'Cluster 2']
Methods.project_onto_R3(cluster_2,[0,1,2])