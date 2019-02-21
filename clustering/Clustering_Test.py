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

df = pd.read_csv(r'MagicTelescope.csv')
df=df.drop(columns=['ID','class:'])

#print(df.mean())
#print(df.std())
df=Methods.cluster_Hierarchical(df,3,'ward',False,True)
#print(Methods.normalize(df))
Methods.cluster_Hierarchical(df,3,'ward',False,True)
#Methods.heatMap(df)

Methods.linkageType(df,'ward')
plt.show()
#Methods.heatMap(data)


#df_DB_w_o = Methods.cluster_DBSCAN(df=df, dim=dim, eps=.4, min_samples=10, keepOutliers=True)
#df_DB = Methods.cluster_DBSCAN(df=df, dim=dim, eps=.4, min_samples=10, keepOutliers=False)
#df_KM = Methods.cluster_KMeans(df=df_DB, dim=dim, k=2)

#Methods.project_onto_R3(df_DB_w_o, [0, 1, 2])
#Methods.project_onto_R3(df_DB, [0, 1, 2])
#Methods.project_onto_R3(df_KM, [0, 1, 2])
#plt.show()


