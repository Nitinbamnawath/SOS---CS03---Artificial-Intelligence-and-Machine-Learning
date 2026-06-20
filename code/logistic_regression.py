"""
Logistic Regression using Gradient Descent
Week 2 - Summer of Science 2026

Dataset: sklearn's make_classification (synthetic, 2 features, 2 classes)
so the decision boundary can be visualized in 2D.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# ---------------------------------------------------------
# 1. Generate dataset
# ---------------------------------------------------------
X, y = make_classification(
    n_samples=200, n_features=2, n_redundant=0,
    n_informative=2, n_clusters_per_class=1, random_state=42
)

# ---------------------------------------------------------
# 2. Sigmoid + Gradient Descent from scratch
# ---------------------------------------------------------
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
print(f"Final weights: {w}")
print(f"Final bias: {b:.4f}")
print(f"Training accuracy: {accuracy * 100:.1f}%")

# ---------------------------------------------------------
# 3. Plot decision boundary
# ---------------------------------------------------------
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = sigmoid(grid @ w + b).reshape(xx.shape)

plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], colors=["#cfe3ff", "#ffd6cf"], alpha=0.6)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="blue", label="Class 0", edgecolor="k")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="green", label="Class 1", edgecolor="k")
plt.contour(xx, yy, probs, levels=[0.5], colors="red", linewidths=2)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"Logistic Regression Decision Boundary (Accuracy: {accuracy*100:.1f}%)")
plt.legend()
plt.tight_layout()
plt.savefig("/home/claude/sos_project/images/logreg_boundary.png", dpi=150)
plt.close()

print("Saved: logreg_boundary.png")
