"""Učitavanje skupova podataka: MNIST, Fashion-MNIST i mali 'digits' (za probu)."""
import numpy as np
from sklearn.datasets import fetch_openml, load_digits

NAZIVI = {
    "mnist": [str(i) for i in range(10)],
    "digits": [str(i) for i in range(10)],
    "fashion": ["Majica", "Pantalone", "Džemper", "Haljina", "Kaput",
                "Sandala", "Košulja", "Patika", "Torba", "Čizma"],
}
OPENML = {"mnist": "mnist_784", "fashion": "Fashion-MNIST"}


def ucitaj_sve(ime):
    """Vraća (X, y, nazivi_klasa, velicina_slike). Pikseli su skalirani u [0, 1]."""
    if ime == "digits":  # 1797 slika 8x8, ugrađeno u scikit-learn
        X, y = load_digits(return_X_y=True)
        return (X / 16.0).astype(np.float32), y.astype(int), NAZIVI[ime], 8

    print(f"Učitavam {OPENML[ime]} sa OpenML-a (prvi put traje duže, posle se čita iz keša)...")
    X, y = fetch_openml(OPENML[ime], version=1, return_X_y=True, as_frame=False)
    return (X / 255.0).astype(np.float32), y.astype(int), NAZIVI[ime], 28


def uzorak(n_ukupno, n, seed=42):
    """Nasumični indeksi bez ponavljanja."""
    rng = np.random.default_rng(seed)
    return rng.choice(n_ukupno, size=min(n, n_ukupno), replace=False)
