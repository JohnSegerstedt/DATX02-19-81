from clustering.Utilities import Methods
import pandas as pd

df = pd.read_csv(r'../data/Replays2-210.0s.csv')
del df['Unnamed: 0']
df = df.loc[:, (df != 0).any(axis=0)]

Methods.explainedVariance(df)

df_cluster=Methods.cluster_KMeans(Methods.clusterPCA(df,2),2,True)
Methods.project_onto_R2(df_cluster,['PCA 1', 'PCA 2'])

df['Names']=df_cluster['Names']

df_highCorr=df[['Probe1','Stalker1', 'Assimilator1', 'CyberneticsCore1', 'Gateway1', 'Nexus1', 'Pylon1', 'Names']]

df=Methods.binaryCluster(df)
#df['Cluster 1'].loc[df['Cluster 1'] == 1] = True
#df.loc[df['Cluster 1'] == 0] = False

#print(df)

Methods.pointBiserial(df,'Cluster 1')
#corr=Methods.heatMap(df)

#Methods.parallelCoordinates(df_highCorr)


