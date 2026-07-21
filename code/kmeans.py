import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

height = [150,152,155,158,160,162,165,168,170,172,175,178,180,182,185,156,163,171,177,183]
weight = [45,48,50,52,55,57,60,63,65,68,72,75,78,80,84,49,58,67,76,82]

X = np.array(list(zip(height, weight)))

wcss = []
for k in range(1, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)

plt.plot(range(1, 7), wcss, marker="o", color="teal")
plt.xlabel("K")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.savefig("/home/claude/sos_project/images/kmeans_elbow.png", dpi=150)
plt.close()

km_final = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = km_final.fit_predict(X)
centers = km_final.cluster_centers_

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", edgecolor="k")
plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=200)
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("K-Means Clustering (K=3)")
plt.savefig("/home/claude/sos_project/images/kmeans_clusters.png", dpi=150)
plt.close()

