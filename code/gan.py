import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("../images", exist_ok=True)
torch.manual_seed(42)

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

latent_dim = 16

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.Sigmoid()
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

G = Generator()
D = Discriminator()
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4)
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4)
criterion = nn.BCELoss()

g_losses, d_losses = [], []

for epoch in range(50):
    g_total, d_total = 0, 0
    for (real,) in loader:
        bs = real.size(0)

        # train discriminator
        opt_D.zero_grad()
        real_labels = torch.ones(bs, 1)
        fake_labels = torch.zeros(bs, 1)
        z = torch.randn(bs, latent_dim)
        fake = G(z).detach()
        d_loss = criterion(D(real), real_labels) + criterion(D(fake), fake_labels)
        d_loss.backward()
        opt_D.step()

        # train generator
        opt_G.zero_grad()
        z = torch.randn(bs, latent_dim)
        fake = G(z)
        g_loss = criterion(D(fake), real_labels)
        g_loss.backward()
        opt_G.step()

        g_total += g_loss.item()
        d_total += d_loss.item()

    g_losses.append(g_total / len(loader))
    d_losses.append(d_total / len(loader))
    if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch+1}/50  G: {g_losses[-1]:.3f}  D: {d_losses[-1]:.3f}")

# plot 1: loss curves
plt.figure(figsize=(7, 4))
plt.plot(g_losses, label="Generator")
plt.plot(d_losses, label="Discriminator")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN Training Losses")
plt.legend()
plt.tight_layout()
plt.savefig("../images/gan_loss.png", dpi=150)
plt.close()

# plot 2: generated images
with torch.no_grad():
    z = torch.randn(16, latent_dim)
    samples = G(z).numpy().reshape(16, 16, 16)

fig, axes = plt.subplots(2, 8, figsize=(10, 3))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i], cmap="gray")
    ax.axis("off")
plt.suptitle("GAN Generated Samples", fontsize=12)
plt.tight_layout()
plt.savefig("../images/gan_output.png", dpi=150)
plt.close()

print("saved gan_loss.png and gan_output.png")

