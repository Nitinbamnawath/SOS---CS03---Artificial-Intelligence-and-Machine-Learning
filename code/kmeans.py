"""
K-Means Clustering + Elbow Method
Week 3 - Summer of Science 2026

Dataset: sklearn's make_blobs (synthetic clustered data).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# 1. Generate dataset
# ---------------------------------------------------------
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

# ---------------------------------------------------------
# 2. Elbow method - WCSS for K = 1 to 10
# ---------------------------------------------------------
wcss = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(K_range, wcss, marker="o", color="teal")
plt.xlabel("Number of clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal K")
plt.tight_layout()
plt.savefig("/home/claude/sos_project/images/kmeans_elbow.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 3. Final clustering with K = 4 (elbow point)
# ---------------------------------------------------------
km_final = KMeans(n_clusters=4, n_init=10, random_state=42)
labels = km_final.fit_predict(X)
centers = km_final.cluster_centers_

plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.7, edgecolor="k")
plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=200, label="Centroids")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering (K=4)")
plt.legend()
plt.tight_layout()
plt.savefig("/home/claude/sos_project/images/kmeans_clusters.png", dpi=150)
plt.close()

print("Saved: kmeans_elbow.png, kmeans_clusters.png")
