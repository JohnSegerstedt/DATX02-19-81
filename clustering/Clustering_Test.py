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
data = data[1:5000, [4, 6, 8]]
dim = len(data[1,:])


#Methods.heatMap(data)



#df_DBSCAN = Methods.cluster_DBSCAN(data=data, dim=11, eps=.4, min_samples=10, outliers=True)

df_KMeans = Methods.cluster_KMeans(data=data, dim=dim, k=3)

#Methods.parallelCoordinates(df_KMeans)







plt.show()


