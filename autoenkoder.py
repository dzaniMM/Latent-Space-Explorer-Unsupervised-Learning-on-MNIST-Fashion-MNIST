"""Autoenkoderi u PyTorch-u.

Za slike 28x28 koristi se konvolucioni autoenkoder, a za ostale (npr. 8x8)
potpuno povezani (MLP). Oba primaju i vraćaju "spljoštene" slike oblika (N, d).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MLPAutoenkoder(nn.Module):
    def __init__(self, d, latent):
        super().__init__()
        h1, h2 = (512, 128) if d >= 512 else (128, 64)
        self.enkoder = nn.Sequential(
            nn.Linear(d, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, latent),
        )
        self.dekoder = nn.Sequential(
            nn.Linear(latent, h2), nn.ReLU(),
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, d), nn.Sigmoid(),     # izlaz u [0, 1], kao pikseli
        )

    def forward(self, x):
        return self.dekoder(self.enkoder(x))


class KonvAutoenkoder(nn.Module):
    """28x28 -> 14x14 -> 7x7 -> latentni vektor -> nazad do 28x28."""

    def __init__(self, latent):
        super().__init__()
        self.enkoder = nn.Sequential(
            nn.Unflatten(1, (1, 28, 28)),
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),   # 14x14
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 7x7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent),
        )
        self.dekoder = nn.Sequential(
            nn.Linear(latent, 64 * 7 * 7), nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.dekoder(self.enkoder(x))


def napravi(vel, latent):
    return KonvAutoenkoder(latent) if vel == 28 else MLPAutoenkoder(vel * vel, latent)


def treniraj(X, vel, latent=10, epohe=20, batch=256, lr=1e-3, seed=0, opis="AE"):
    torch.manual_seed(seed)
    uredjaj = "cuda" if torch.cuda.is_available() else "cpu"
    model = napravi(vel, latent).to(uredjaj)
    loader = DataLoader(TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32))),
                        batch_size=batch, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"  [{opis}] treniranje na {len(X)} slika, latent={latent}, uređaj={uredjaj}")
    for e in range(epohe):
        model.train()
        ukupno = 0.0
        for (xb,) in loader:
            xb = xb.to(uredjaj)
            opt.zero_grad()
            gubitak = F.mse_loss(model(xb), xb)
            gubitak.backward()
            opt.step()
            ukupno += gubitak.item() * len(xb)
        if e == 0 or (e + 1) % 5 == 0 or e == epohe - 1:
            print(f"    epoha {e + 1:>3}/{epohe}  MSE={ukupno / len(X):.5f}")
    model.eval()
    model.uredjaj = uredjaj
    return model


@torch.no_grad()
def _u_serijama(fn, model, X, batch=1024):
    izlazi = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.asarray(X[i:i + batch], dtype=np.float32)).to(model.uredjaj)
        izlazi.append(fn(xb).cpu().numpy())
    return np.concatenate(izlazi)


def kodiraj(model, X):
    """Latentna reprezentacija (izlaz enkodera)."""
    return _u_serijama(model.enkoder, model, X)


def rekonstruisi(model, X):
    return _u_serijama(model, model, X)
