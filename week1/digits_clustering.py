
from sklearn.datasets import load_digits
digits = load_digits()
# since there are 10 digits we make 10 clusters
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(digits.data)

# visualize the cluster centers, the cluster centers look like the digits
fig = plt.figure(figsize=(8, 3))
for i in range(10):
    ax = fig.add_subplot(2, 5, 1 + i)
    ax.imshow(kmeans.cluster_centers_[i].reshape((8, 8)),
              cmap=plt.cm.binary)