from clustering.Utilities import Methods
import pandas as pd
import numpy as np
from clustering.Utilities.som import SOM
from minisom import MiniSom

df = pd.read_csv(r'C:\Users\arvid\Desktop\Skola\Skolår 3\Kandidatarbete\iris.csv')
df = Methods.clusterPCA(df, 2)
df = df.drop('PC 2', axis=1)
print(np.sum(df.var().values))
df_kmeans = Methods.cluster_KMeans(df, 3, True)
print(np.sum(df_kmeans.loc[df_kmeans['Names'] == 'Cluster 1'].var().values))
print(np.sum(df_kmeans.loc[df_kmeans['Names'] == 'Cluster 2'].var().values))
print(np.sum(df_kmeans.loc[df_kmeans['Names'] == 'Cluster 3'].var().values))

N1 = list(df['Names'])
N2 = list(df_kmeans['Names'])
right = 0
for i in range(0, len(list(df['Names'])), 1):
    if N1[i] == 'Iris-setosa' and N2[i] == 'Cluster 2':
        right +=1
    elif N1[i] == 'Iris-virginica' and N2[i] == 'Cluster 3':
        right += 1
    elif N1[i] == 'Iris-versicolor' and N2[i] == 'Cluster 1':
        right += 1



print(right)