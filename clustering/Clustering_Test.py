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



df = Methods.cluster_DBSCAN(data=data, dim=dim, eps=.4, min_samples=10, outliers=True)

#df = Methods.cluster_KMeans(data=data, dim=dim, k=2)

Methods.project_onto_R3(df, [0, 1, 2])






plt.show()


