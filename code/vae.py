import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("../images", exist_ok=True)
torch.manual_seed(42)

# synthetic dataset: 2000 small 16x16 grayscale images with random blobs
# stands in for MNIST since download is unavailable in this environment
def make_dataset(n=2000):
    data = []
    for _ in range(n):
        img = np.zeros((16, 16), dtype=np.float32)
        cx, cy = np.random.randint(4, 12, 2)
        r = np.random.randint(2, 5)
        for i in range(16):
            for j in range(16):
                if (i - cx)**2 + (j - cy)**2 < r**2:
                    img[i, j] = 1.0
        data.append(img.flatten())
    return torch.tensor(np.array(data))

X = make_dataset()
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X),
    batch_size=64, shuffle=True
)

latent_dim = 8

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, 256)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return torch.sigmoid(self.fc4(F.relu(self.fc3(z))))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []
for epoch in range(30):
    total = 0
    for (x,) in loader:
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl
        loss.backward()
        optimizer.step()
        total += loss.item()
    losses.append(total / len(X))
    print(f"epoch {epoch+1}/30  loss: {losses[-1]:.2f}")

# plot 1: loss curve
plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("ELBO Loss")
plt.title("VAE Training Loss")
plt.tight_layout()
plt.savefig("../images/vae_loss.png", dpi=150)
plt.close()

# plot 2: generated samples
with torch.no_grad():
    z = torch.randn(16, latent_dim)
    samples = model.decode(z).numpy().reshape(16, 16, 16)

fig, axes = plt.subplots(2, 8, figsize=(10, 3))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i], cmap="gray")
    ax.axis("off")
plt.suptitle("VAE Generated Samples", fontsize=12)
plt.tight_layout()
plt.savefig("../images/vae_output.png", dpi=150)
plt.close()

print("saved vae_loss.png and vae_output.png")

