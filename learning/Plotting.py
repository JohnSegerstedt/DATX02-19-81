import numpy as np
from matplotlib import pyplot as plt


x1 = np.load('Frames_RF_Frames.npy')
y1 = np.load('accuracy_RF_Frames.npy')


plt.scatter(x1, y1, color='red', marker=">", label='RF Accuracy')
plt.plot(x1,y1, '-', color = 'red', lw=2)
plt.plot([0,20000],[1/5,1/5], 'r:', lw=2, label='Chance')
plt.plot([0,20000],[.6,.6], 'g-.', lw=2, label='Baseline')
plt.plot([5760,5760],[-3,3], 'm--', lw=2, label='Clustering timestamp')
plt.axis([0,20000,0,1.2])
plt.xlabel('Data until Frame')
plt.ylabel('Model Accuracy')
plt.title('Model Accuracy vs. frames trained on')
plt.legend(loc='lower right')
plt.show()


print()
