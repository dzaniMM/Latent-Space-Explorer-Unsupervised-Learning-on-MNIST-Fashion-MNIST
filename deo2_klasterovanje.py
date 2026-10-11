"""DEO 2: Klasterovanje (K-means, GMM, aglomerativno) u različitim prostorima."""
import time

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph

from grafici import rasipanje, sacuvaj, plt
from metrike import sve_mere

PROSTORI = ["PCA-50", "UMAP-10", "AE-10", "AE-UMAP-10"]


def ward(P, k):
    """Klasičan Ward pamti matricu rastojanja n x n (za 70000 slika ~39 GB).
    Za veće skupove spajanje se ograničava na susede u grafu 15 najbližih
    suseda, što radi u par sekundi i malo memorije."""
    graf = kneighbors_graph(P, 15, include_self=False) if len(P) > 15000 else None
    return AgglomerativeClustering(k, linkage="ward", connectivity=graf).fit_predict(P)


def algoritmi(k=10, seed=0):
    return {
        "K-means": lambda P: KMeans(k, n_init=10, random_state=seed).fit_predict(P),
        "GMM": lambda P: GaussianMixture(k, covariance_type="full", n_init=2,
                                         random_state=seed).fit(P).predict(P),
        "Aglomerativno": lambda P: ward(P, k),
    }


def pokreni(X, y, p, d, args):
    print("\n" + "=" * 60 + "\nDEO 2: KLASTEROVANJE\n" + "=" * 60)
    alg = algoritmi()
    redovi, oznake = [], {}

    def uradi(ime_p, P, ime_a, fn):
        t = time.time()
        kl = fn(P)
        r = {"prostor": ime_p, "algoritam": ime_a, **sve_mere(y, kl), "vreme_s": time.time() - t}
        print(f"  {ime_p:<15}{ime_a:<15} tačnost={r['tacnost']:.3f} ARI={r['ARI']:.3f} "
              f"klastera={r['br_klastera']} "
              f"({r['vreme_s']:.1f}s)")
        redovi.append(r)
        oznake[f"{ime_p}|{ime_a}"] = kl

    # Polazna tačka: K-means direktno na pikselima
    uradi("sirovi pikseli", X, "K-means", alg["K-means"])
    for ime_p in PROSTORI:
        for ime_a, fn in alg.items():
            uradi(ime_p, p[ime_p], ime_a, fn)

    df = pd.DataFrame(redovi)
    df.to_csv(d / "2_rezultati_klasterovanja.csv", index=False, float_format="%.4f")
    np.savez(d / "2_klasteri.npz", **oznake)
    print(f"    -> {d / '2_rezultati_klasterovanja.csv'}")

    # Toplotne mape: prostor x algoritam
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, mera in zip(axes, ["tacnost", "ARI"]):
        tab = df[df.prostor != "sirovi pikseli"].pivot(index="prostor", columns="algoritam", values=mera)
        tab = tab.reindex(index=PROSTORI)
        ax.imshow(tab.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        for i in range(tab.shape[0]):
            for j in range(tab.shape[1]):
                v = tab.values[i, j]
                ax.text(j, i, "-" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                        color="white" if (np.isnan(v) or v < 0.6) else "black")
        ax.set_xticks(range(tab.shape[1]), tab.columns)
        ax.set_yticks(range(tab.shape[0]), tab.index)
        bazna = df[df.prostor == "sirovi pikseli"][mera].iloc[0]
        ax.set_title(f"{ {'tacnost': 'Tačnost'}.get(mera, mera)} (K-means na pikselima: {bazna:.2f})")
    sacuvaj(fig, d / "2a_toplotna_mapa.png")

    # Isti 2D UMAP prikaz, obojen rezultatima svake kombinacije
    imena_a = list(alg)
    fig, axes = plt.subplots(len(imena_a), len(PROSTORI), figsize=(5 * len(PROSTORI), 4.5 * len(imena_a)))
    for i, ime_a in enumerate(imena_a):
        for j, ime_p in enumerate(PROSTORI):
            ax = axes[i, j]
            kljuc = f"{ime_p}|{ime_a}"
            if kljuc in oznake:
                r = df[(df.prostor == ime_p) & (df.algoritam == ime_a)].iloc[0]
                rasipanje(ax, p["2D_UMAP"], oznake[kljuc],
                          f"{ime_a} na {ime_p}\ntačnost={r.tacnost:.2f}, klastera={r.br_klastera}")
            else:
                ax.axis("off")
    sacuvaj(fig, d / "2b_klasteri_na_umap_mapi.png")
    return df, oznake
