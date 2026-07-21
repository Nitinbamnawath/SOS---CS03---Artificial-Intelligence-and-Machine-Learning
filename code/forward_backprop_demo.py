import numpy as np

# tiny 2-layer network, worked by hand to see backprop concretely
np.random.seed(0)

# 1 training example, 2 input features, 1 hidden layer of 3 units, 1 output
x = np.array([[0.5, 0.8]])          # shape (1, 2)
y = np.array([[1.0]])               # true label

W1 = np.random.randn(2, 3) * 0.5    # input -> hidden
b1 = np.zeros((1, 3))
W2 = np.random.randn(3, 1) * 0.5    # hidden -> output
b2 = np.zeros((1, 1))

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

lr = 0.1
for step in range(500):
    # forward pass
    z1 = x @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)          # final prediction

    loss = -(y * np.log(a2) + (1 - y) * np.log(1 - a2)).mean()

    # backward pass (chain rule, layer by layer)
    dz2 = a2 - y                        # dL/dz2 for BCE + sigmoid simplifies nicely
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = da1 * relu_deriv(z1)
    dW1 = x.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)

    # gradient descent update, same rule as every other model in this report
    W2 -= lr * dW2; b2 -= lr * db2
    W1 -= lr * dW1; b1 -= lr * db1

    if step % 100 == 0:
        print(f"step {step:3d}  loss {loss:.4f}  prediction {a2[0,0]:.4f}")

print("final prediction:", a2[0, 0], " target:", y[0, 0])
