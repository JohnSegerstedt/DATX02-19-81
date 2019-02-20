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

df = pd.DataFrame({'Names': 'foo bar foo bar foo bar foo foo'.split(),
                   'B': 'one one two three two two one three'.split(),
                   'C': np.arange(8), 'D': np.arange(8) * 2})
print(df)
dfNew=Methods.pickOutCluster(df,'foo')
print(dfNew)

df = pd.read_csv(r'MagicTelescope.csv')
print(df.columns)

df=df.drop(columns=['ID','class:'])

df=Methods.cluster_KMeans2(df,3,False,True)

Methods.heatMap(df)



#Methods.heatMap(data)


#df_DB_w_o = Methods.cluster_DBSCAN(df=df, dim=dim, eps=.4, min_samples=10, keepOutliers=True)
#df_DB = Methods.cluster_DBSCAN(df=df, dim=dim, eps=.4, min_samples=10, keepOutliers=False)
#df_KM = Methods.cluster_KMeans(df=df_DB, dim=dim, k=2)

#Methods.project_onto_R3(df_DB_w_o, [0, 1, 2])
#Methods.project_onto_R3(df_DB, [0, 1, 2])
#Methods.project_onto_R3(df_KM, [0, 1, 2])
#plt.show()


