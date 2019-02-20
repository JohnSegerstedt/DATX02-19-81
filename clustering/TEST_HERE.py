'''
plt.subplot(2,2,3)
plt.title('Parallel Coordinates K-Means clustering')
pd.plotting.parallel_coordinates(
        df_KMeans, 'Names', color=color)

plt.subplot(2,4,3)
singleLinkage(A)

plt.subplot(2,4,4)
completeLinkage(A)

plt.subplot(2,4,7)
wardLinkage(A)


plt.subplot(2,4,8)
averageLinkage(A)
'''