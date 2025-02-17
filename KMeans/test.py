from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X_train, y_train = make_blobs(n_samples=100, centers=3, n_features=2)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)