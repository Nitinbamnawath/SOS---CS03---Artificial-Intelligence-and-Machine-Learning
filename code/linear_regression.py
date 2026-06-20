"""
Simple Linear Regression using Gradient Descent
Week 2 - Summer of Science 2026

Dataset: synthetic data resembling CGPA vs Package (LPA), generated with
numpy for reproducibility (random_state fixed). This mirrors the structure
of the example dataset shown in the mentor's reference notes.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Generate dataset
# ---------------------------------------------------------
np.random.seed(42)
n = 200
cgpa = np.random.uniform(4.5, 9.5, n)
package = 0.55 * cgpa - 0.2 + np.random.normal(0, 0.35, n)  # roughly linear with noise

# ---------------------------------------------------------
# 2. Gradient Descent from scratch
# ---------------------------------------------------------
w, b = 0.0, 0.0
alpha = 0.01
epochs = 2000
m = len(cgpa)

cost_history = []

for epoch in range(epochs):
    y_hat = w * cgpa + b
    cost = (1 / (2 * m)) * np.sum((y_hat - package) ** 2)
    cost_history.append(cost)

    dw = (1 / m) * np.sum((y_hat - package) * cgpa)
    db = (1 / m) * np.sum(y_hat - package)

    w -= alpha * dw
    b -= alpha * db

print(f"Final w (slope): {w:.4f}")
print(f"Final b (intercept): {b:.4f}")
print(f"Final cost: {cost_history[-1]:.4f}")

# ---------------------------------------------------------
# 3. Plot 1: Fit line over data
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(cgpa, package, alpha=0.5, color="royalblue", label="Data")
x_line = np.linspace(cgpa.min(), cgpa.max(), 100)
plt.plot(x_line, w * x_line + b, color="red", linewidth=2, label="Fitted line")
plt.xlabel("CGPA")
plt.ylabel("Package (LPA)")
plt.title("Simple Linear Regression - Gradient Descent Fit")
plt.legend()
plt.tight_layout()
plt.savefig("/home/claude/sos_project/images/linreg_fit.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 4. Plot 2: Cost over epochs
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(cost_history, color="darkorange")
plt.xlabel("Epoch")
plt.ylabel("Cost (MSE)")
plt.title("Cost vs Epoch during Gradient Descent")
plt.tight_layout()
plt.savefig("/home/claude/sos_project/images/linreg_cost.png", dpi=150)
plt.close()

print("Saved: linreg_fit.png, linreg_cost.png")
