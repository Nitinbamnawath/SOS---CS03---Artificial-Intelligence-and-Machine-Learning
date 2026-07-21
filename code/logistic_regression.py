import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=200, n_features=2, n_redundant=0,
    n_informative=2, n_clusters_per_class=1, random_state=42
)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w = np.zeros(X.shape[1])
b = 0.0
alpha = 0.1
epochs = 2000
m = len(y)

for epoch in range(epochs):
    z = X @ w + b
    y_hat = sigmoid(z)

    dw = (1 / m) * X.T @ (y_hat - y)
    db = (1 / m) * np.sum(y_hat - y)

    w -= alpha * dw
    b -= alpha * db

predictions = (sigmoid(X @ w + b) >= 0.5).astype(int)
accuracy = np.mean(predictions == y)
print(f"Training accuracy: {accuracy * 100:.1f}%")

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = sigmoid(grid @ w + b).reshape(xx.shape)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
plt.contour(xx, yy, probs, levels=[0.5], colors="black")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistic Regression Decision Boundary")
plt.savefig("logreg_boundary.png")
plt.show()

