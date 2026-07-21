import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

np.random.seed(0)

digits = load_digits()
X = digits.images / 16.0          # normalize to [0, 1], shape (n, 8, 8)
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

n_filters, fsize = 8, 3
W_conv = np.random.randn(n_filters, fsize, fsize) * 0.3
b_conv = np.zeros(n_filters)

conv_out = 8 - fsize + 1          # 6
pool_out = conv_out // 2          # 3
flat_size = n_filters * pool_out * pool_out
n_classes = 10

W_fc = np.random.randn(flat_size, n_classes) * (1.0 / np.sqrt(flat_size))
b_fc = np.zeros(n_classes)

def relu(z): return np.maximum(0, z)
def relu_deriv(z): return (z > 0).astype(float)

def conv_forward(img, W, b):
    out = np.zeros((n_filters, conv_out, conv_out))
    for f in range(n_filters):
        for i in range(conv_out):
            for j in range(conv_out):
                patch = img[i:i+fsize, j:j+fsize]
                out[f, i, j] = np.sum(patch * W[f]) + b[f]
    return out

def maxpool_forward(a):
    out = np.zeros((n_filters, pool_out, pool_out))
    argmax = np.zeros((n_filters, pool_out, pool_out, 2), dtype=int)
    for f in range(n_filters):
        for i in range(pool_out):
            for j in range(pool_out):
                window = a[f, 2*i:2*i+2, 2*j:2*j+2]
                idx = np.unravel_index(np.argmax(window), window.shape)
                argmax[f, i, j] = idx
                out[f, i, j] = window[idx]
    return out, argmax

def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)

def forward(img):
    z_conv = conv_forward(img, W_conv, b_conv)
    a_conv = relu(z_conv)
    a_pool, argmax = maxpool_forward(a_conv)
    flat = a_pool.reshape(-1)
    logits = flat @ W_fc + b_fc
    probs = softmax(logits)
    cache = (img, z_conv, a_conv, a_pool, argmax, flat)
    return probs, cache

def backward(probs, y_true, cache, lr):
    global W_conv, b_conv, W_fc, b_fc
    img, z_conv, a_conv, a_pool, argmax, flat = cache
    y_onehot = np.zeros(n_classes); y_onehot[y_true] = 1
    dlogits = probs - y_onehot

    dW_fc = np.outer(flat, dlogits)
    db_fc = dlogits
    dflat = W_fc @ dlogits
    dpool = dflat.reshape(n_filters, pool_out, pool_out)

    dconv_relu = np.zeros_like(a_conv)
    for f in range(n_filters):
        for i in range(pool_out):
            for j in range(pool_out):
                di, dj = argmax[f, i, j]
                dconv_relu[f, 2*i+di, 2*j+dj] = dpool[f, i, j]
    dz_conv = dconv_relu * relu_deriv(z_conv)

    dW_conv = np.zeros_like(W_conv)
    db_conv = np.zeros_like(b_conv)
    for f in range(n_filters):
        for i in range(conv_out):
            for j in range(conv_out):
                patch = img[i:i+fsize, j:j+fsize]
                dW_conv[f] += dz_conv[f, i, j] * patch
                db_conv[f] += dz_conv[f, i, j]

    W_fc -= lr * dW_fc; b_fc -= lr * db_fc
    W_conv -= lr * dW_conv; b_conv -= lr * db_conv

def loss_fn(probs, y_true):
    return -np.log(probs[y_true] + 1e-9)

lr = 0.05
batch_size = 32
epochs = 40
n_train = len(X_train)

epoch_losses, epoch_accs = [], []
for epoch in range(epochs):
    perm = np.random.permutation(n_train)
    total_loss, correct = 0.0, 0
    for start in range(0, n_train, batch_size):
        idx = perm[start:start+batch_size]
        for i in idx:
            probs, cache = forward(X_train[i])
            total_loss += loss_fn(probs, y_train[i])
            if np.argmax(probs) == y_train[i]:
                correct += 1
            backward(probs, y_train[i], cache, lr)
    epoch_losses.append(total_loss / n_train)
    epoch_accs.append(correct / n_train)
    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f"epoch {epoch:2d}  loss {epoch_losses[-1]:.4f}  train_acc {epoch_accs[-1]*100:.1f}%")

# test evaluation
test_preds = []
for i in range(len(X_test)):
    probs, _ = forward(X_test[i])
    test_preds.append(np.argmax(probs))
test_preds = np.array(test_preds)
test_acc = accuracy_score(y_test, test_preds)
print(f"final test accuracy: {test_acc*100:.1f}%")

plt.figure(figsize=(6, 4))
plt.plot(epoch_losses, color="darkred")
plt.xlabel("Epoch"); plt.ylabel("Avg. training loss")
plt.title("From-scratch NumPy CNN: Training Loss")
plt.tight_layout()
plt.savefig("../images/cnn_loss.png", dpi=150)
plt.close()

cm = confusion_matrix(y_test, test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Purples", colorbar=False)
plt.title(f"CNN Confusion Matrix (Test Accuracy: {test_acc*100:.1f}%)")
plt.tight_layout()
plt.savefig("../images/cnn_confusion.png", dpi=150)
plt.close()

fig, axes = plt.subplots(2, 8, figsize=(12, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i], cmap="gray")
    color = "green" if test_preds[i] == y_test[i] else "red"
    ax.set_title(f"pred:{test_preds[i]}/true:{y_test[i]}", color=color, fontsize=9)
    ax.axis("off")
plt.suptitle("CNN Sample Predictions", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("../images/cnn_predictions.png", dpi=150)
plt.close()

print("saved cnn_loss.png, cnn_confusion.png, cnn_predictions.png")
