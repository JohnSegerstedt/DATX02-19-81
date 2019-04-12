import numpy as np
from matplotlib import pyplot as plt


x1 = np.load('ass2.npy')
y1 = np.load('succ2.npy')
x2 = np.load('ass3.npy')
y2 = np.load('succ3.npy')
x3 = np.load('pvec100.npy')
y3 = np.load('accuracy100.npy')


plt.plot(x1, y1, 'b-.', lw=2, label='RF Accuracy')
plt.plot(x2, y2, '--', lw=2, label='K-NN Accuracy (K = 7)')
plt.plot(x3, y3, '-', lw=2, label='ANN Accuracy (Dense layers)')
plt.plot([0,1],[1/5,1/5], 'r:', lw=2, label='Chance')
plt.axis([0,1,0,1.2])
plt.xlabel('p(mislabeling)')
plt.ylabel('Model Accuracy')
plt.title('Model Accuracy vs. mislabeling probability')
plt.legend(loc='upper right')
plt.show()


print()
