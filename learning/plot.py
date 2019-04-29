import matplotlib.pyplot as plt
import numpy as np

def accuracyfor5760():
    x = [0, 720, 1440, 2160, 2880, 3600, 4320, 5040, 5760, 6480, 7200, 7920, 8640]
    yfull = [60, 60, 74.97, 77.76, 78.35, 88.26, 93.62, 96.93, 97.46, 96.83, 95.67, 96.39, 96.44]
    ytrain =[60, 60, 68.74, 75.92, 77.23, 85.95, 89.01, 90.33, 91.00, 91.88, 93.30, 91.68, 92.36]
    yp1 = [60.20, 60.17, 65.89, 69.54, 71.34, 79.32, 83.22, 84.23, 84.79, 84.53, 84.19, 83.86, 84.83]
    plt.plot(x, yfull, 'r')
    plt.plot(x, ytrain, 'b')
    plt.plot(x, yp1, 'g')
    plt.axvline(x=5760)
    plt.show()

def plot_accuracy_per_time():
    x = [720, 1440, 2160, 2880, 3600, 4320, 5040, 5760, 6480, 7200, 7920, 8640, 9360, 10080, 10800, 11520, 12240, 12960, 13680, 14400, 15120, 15840, 16560, 17280, 18000, 18720, 19440, 20160, 20880]
    accuracy = [99.98, 99.98, 99.89, 99.57, 99.58, 98.82, 98.97, 97.22, 96.23, 97.07, 91.17, 95.39, 90.89, 96.05, 87.55, 96.69, 81.36, 82.72, 77.13, 73.44, 66.39, 81.78, 95.76, 69.83, 74.73, 81.50, 78.06, 91.24, 87.95]
    baseline = [98.18, 72.59, 73.09, 74.95, 77.74, 63.94, 65.56, 60.20, 58.18, 60.61, 25.48, 61.03, 41.88, 69.31, 30.94, 70.28, 32.49, 33.09, 24.05, 22.38, 29.17, 45.34, 78.38, 28.62, 47.64, 37.55, 44.14, 61.08, 70.28]
    clusters = [2, 3, 3, 3, 3, 5, 5, 6, 6, 6, 9, 5, 8, 5, 7, 4, 8, 8, 8, 8, 9, 6, 3, 7, 5, 5, 5, 4, 4]
    #plt.plot(x, accuracy, 'r')
    #plt.plot(x, baseline, 'b')
    #plt.plot(x, clusters, 'g')

    fig, ax1 = plt.subplots()
    color = 'brown'
    ax1.set_xlabel('Time [s]', fontsize=14)
    ax1.set_ylabel('Model accuracy above baseline', color=color, fontsize=14)
    accuracy = np.array(accuracy)
    baseline = np.array(baseline)
    x = np.array(x)
    clusters = np.array(clusters)

    x = x/24
    accuracy = accuracy/100
    baseline = baseline/100

    ax1.scatter(x, np.divide((accuracy - baseline), (np.ones(len(x)) - baseline)),
                color=color, marker='>')
    ax1.legend(['Model accuracy above baseline'])
    ax1.axis([0, 23000/24, 0, 1.2])
    ax2 = ax1.twinx()
    color = 'darkcyan'
    ax2.set_ylabel('Optimal number of clusters', color=color, fontsize=14)
    ax2.scatter(x, clusters, color=color, marker='x')
    ax2.legend(['Optimal number of clusters'], loc='lower right')
    ax2.axis([0, 21000/24, 0, 10.2])
    fig.tight_layout()
    plt.title('Model accuracy vs. provided data for optimal clustering')

    plt.show()

if __name__ == "__main__":
   plot_accuracy_per_time()