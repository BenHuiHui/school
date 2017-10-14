from sklearn.cluster import KMeans
import numpy as np

x = np.array([
    [2,2], [3,2],[4,2],[5,2],
    [3,3], [4,3], [7,3],
    [3,4],
    [2,5], [3,5],
    [3,6], [4,6], [7,6],
    [3,7],[6,7],[7,7], [8,7],
    [7,8]
])

kmeans = KMeans(n_clusters=3, init=np.array([
    [2,1], [3,8], [9,9]
]), max_iter=3).fit(x)

print kmeans.cluster_centers_
print kmeans.labels_


