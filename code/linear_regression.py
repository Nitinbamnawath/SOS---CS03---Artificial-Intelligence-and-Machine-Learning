import numpy as np
import matplotlib.pyplot as plt

hours = np.array([1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,2,3.5,5,6.5,8])
marks = np.array([35,40,42,48,50,55,58,62,65,68,72,75,78,80,85,38,52,67,76,88])

w, b = 0.0, 0.0
alpha = 0.01
epochs = 2000
n = len(hours)
cost_history = []

for i in range(epochs):
    y_hat = w * hours + b
    cost = np.mean((y_hat - marks) ** 2) / 2
    cost_history.append(cost)
    dw = np.mean((y_hat - marks) * hours)
    db = np.mean(y_hat - marks)
    w -= alpha * dw
    b -= alpha * db

print("slope:", w, "intercept:", b)

plt.scatter(hours, marks, color="royalblue")
x_line = np.linspace(hours.min(), hours.max(), 50)
plt.plot(x_line, w * x_line + b, color="red")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.savefig("linreg_fit.png")
plt.close()

plt.plot(cost_history, color="darkorange")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Cost vs Epoch")
plt.savefig("linreg_cost2.png")
plt.close()
