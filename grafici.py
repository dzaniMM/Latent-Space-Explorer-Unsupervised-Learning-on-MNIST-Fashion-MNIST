"""Pomoćne funkcije za crtanje."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix


def sacuvaj(fig, putanja):
    fig.tight_layout()
    fig.savefig(putanja, dpi=120)
    plt.close(fig)
    print(f"    -> {putanja}")


def rasipanje(ax, Z, oznake, naslov, nazivi=None, s=2):
    """2D prikaz tačaka obojenih po oznaci; oznaka -1 (šum) je siva."""
    oznake = np.asarray(oznake)
    jedinstvene = np.unique(oznake[oznake >= 0])
    cmap = plt.get_cmap("tab10" if len(jedinstvene) <= 10 else "tab20")
    sum_maska = oznake < 0
    if sum_maska.any():
        ax.scatter(Z[sum_maska, 0], Z[sum_maska, 1], c="lightgray", s=s, alpha=0.4,
                   label="šum" if nazivi else None)
    for i, o in enumerate(jedinstvene):
        m = oznake == o
        ax.scatter(Z[m, 0], Z[m, 1], color=cmap(i % cmap.N), s=s, alpha=0.7,
                   label=nazivi[o] if nazivi else None)
    if nazivi:
        ax.legend(fontsize=6, markerscale=3, loc="best")
    ax.set_title(naslov, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def mreza_slika(slike, vel, naslovi=None, kolone=10, naslov=None, boje=None, cmap="gray_r"):
    n = len(slike)
    redovi = int(np.ceil(n / kolone))
    fig, axes = plt.subplots(redovi, kolone, figsize=(1.3 * kolone, 1.5 * redovi), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(np.asarray(slike[i]).reshape(vel, vel), cmap=cmap)
            if naslovi is not None:
                ax.set_title(naslovi[i], fontsize=7, color=(boje[i] if boje else "black"))
    if naslov:
        fig.suptitle(naslov)
    return fig


def matrica_klastera(y, kl, nazivi, naslov, putanja):
    """Klase (redovi) x klasteri (kolone), kolone poređane tako da dijagonala
    odgovara najboljem uparivanju."""
    klasteri = np.unique(kl)
    cm = contingency_matrix(y, kl)
    validni = klasteri >= 0
    cmv, kv = cm[:, validni], klasteri[validni]
    r, c = linear_sum_assignment(-cmv)
    redosled = list(c[np.argsort(r)]) + [j for j in range(len(kv)) if j not in set(c)]
    kolone = [cmv[:, j] for j in redosled]
    etikete = [f"K{kv[j]}" for j in redosled]
    if (~validni).any():
        kolone.append(cm[:, ~validni].sum(axis=1)); etikete.append("šum")
    M = np.stack(kolone, axis=1)

    fig, ax = plt.subplots(figsize=(max(6, 0.55 * M.shape[1] + 2), 6))
    ax.imshow(M, cmap="Blues", aspect="auto")
    if M.shape[1] <= 25:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, M[i, j], ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] > M.max() / 2 else "black")
    ax.set_yticks(range(M.shape[0]), [nazivi[k] for k in np.unique(y)])
    ax.set_xticks(range(M.shape[1]), etikete, rotation=90, fontsize=7)
    ax.set_ylabel("Prava klasa"); ax.set_xlabel("Pronađeni klaster")
    ax.set_title(naslov, fontsize=10)
    sacuvaj(fig, putanja)
