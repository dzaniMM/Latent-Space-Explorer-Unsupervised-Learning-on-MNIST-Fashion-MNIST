"""DEO 1: Smanjenje dimenzionalnosti (PCA, t-SNE, UMAP, autoenkoder)."""
import time

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import autoenkoder as ae
from grafici import mreza_slika, rasipanje, sacuvaj, plt


def pca_analiza(X, vel, d):
    print("\n[1a] PCA: objašnjena varijansa, glavne komponente, rekonstrukcije")
    pca = PCA().fit(X)
    kum = np.cumsum(pca.explained_variance_ratio_)
    pragovi = {p: int(np.searchsorted(kum, p) + 1) for p in (0.80, 0.90, 0.95, 0.99)}
    for p, k in pragovi.items():
        print(f"  {p:.0%} varijanse -> {k} od {X.shape[1]} komponenti")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(kum) + 1), kum)
    for p, k in pragovi.items():
        ax.axhline(p, ls=":", c="gray", lw=0.8)
        ax.annotate(f"{p:.0%}: {k}", (k, p), fontsize=7, xytext=(5, -10), textcoords="offset points")
    ax.set_xlabel("Broj komponenti"); ax.set_ylabel("Kumulativna objašnjena varijansa")
    ax.set_title("PCA: koliko informacija nosi koliko komponenti")
    sacuvaj(fig, d / "1a_pca_varijansa.png")

    # Glavne komponente prikazane kao slike ("eigendigits")
    komp = pca.components_[:10]
    fig = mreza_slika(komp, vel, [f"PC{i + 1}" for i in range(10)], cmap="RdBu_r",
                      naslov="Prvih 10 glavnih komponenti")
    sacuvaj(fig, d / "1b_glavne_komponente.png")

    # Rekonstrukcija sa različitim brojem komponenti
    primeri = X[:8]
    ks = [k for k in (2, 5, 10, 20, 50, 150) if k < X.shape[1]]
    redovi, naslovi = [primeri], ["original"] * 8
    for k in ks:
        p = PCA(n_components=k).fit(X)
        redovi.append(p.inverse_transform(p.transform(primeri)))
        naslovi += [f"k={k}"] * 8
    fig = mreza_slika(np.concatenate(redovi), vel, naslovi, kolone=8,
                      naslov="PCA rekonstrukcija sa k komponenti")
    sacuvaj(fig, d / "1c_pca_rekonstrukcije.png")
    return {f"pca_komp_{int(p * 100)}%": k for p, k in pragovi.items()}


def izracunaj_prostore(X, vel, d, epohe, susedi=30, seed=0):

    kes = d / "prostori.npz"
    if kes.exists():
        z = np.load(kes, allow_pickle=True)
        if (int(z["n"]) == len(X) and int(z["epohe"]) == epohe
                and "susedi" in z.files and int(z["susedi"]) == susedi):
            print("\n[1d] Prostori učitani iz keša (obriši prostori.npz za ponovno računanje)")
            return ({k: z[k] for k in z.files if k not in ("n", "epohe", "susedi", "info")},
                    z["info"].item())

    print("\n[1d] Računam reprezentacije: PCA-50, UMAP-10, autoenkoder-10, AE-UMAP-10, 2D prikazi")
    p, info = {}, {}
    n_pca = min(50, X.shape[1])
    t = time.time(); p["PCA-50"] = PCA(n_components=n_pca, random_state=seed).fit_transform(X)
    info["vreme_PCA_s"] = time.time() - t

    # Za klasterovanje: više dimenzija i min_dist=0 daju kompaktnije grupe
    t = time.time()
    p["UMAP-10"] = umap.UMAP(n_components=10, n_neighbors=susedi, min_dist=0.0,
                             random_state=seed).fit_transform(X)
    info["vreme_UMAP_s"] = time.time() - t

    t = time.time()
    model = ae.treniraj(X, vel, latent=10, epohe=epohe, seed=seed, opis="AE ceo skup")
    # standardizacija: latentne dimenzije mogu imati veoma različite skale,
    # a K-means i Ward su osetljivi na to
    p["AE-10"] = StandardScaler().fit_transform(ae.kodiraj(model, X))
    info["vreme_AE_s"] = time.time() - t

    t = time.time()
    p["AE-UMAP-10"] = umap.UMAP(n_components=10, n_neighbors=susedi, min_dist=0.0,
                                random_state=seed).fit_transform(p["AE-10"])
    info["vreme_AE_UMAP_s"] = time.time() - t

    # Pošteno poređenje: AE sa 10 latentnih dimenzija vs PCA sa 10 komponenti
    pca10 = PCA(n_components=10).fit(X)
    rek_pca = pca10.inverse_transform(pca10.transform(X))
    rek_ae = ae.rekonstruisi(model, X)
    info["MSE_rekonstrukcije_PCA10"] = float(((rek_pca - X) ** 2).mean())
    info["MSE_rekonstrukcije_AE10"] = float(((rek_ae - X) ** 2).mean())
    print(f"  MSE rekonstrukcije sa 10 dimenzija: PCA={info['MSE_rekonstrukcije_PCA10']:.4f}, "
          f"AE={info['MSE_rekonstrukcije_AE10']:.4f}")
    fig = mreza_slika(np.concatenate([X[:10], rek_pca[:10], rek_ae[:10]]), vel,
                      ["original"] * 10 + ["PCA-10"] * 10 + ["AE-10"] * 10,
                      naslov="Rekonstrukcija iz 10 dimenzija: PCA vs autoenkoder")
    sacuvaj(fig, d / "1e_pca_vs_ae_rekonstrukcija.png")

    # 2D prikazi
    print("  2D projekcije (t-SNE može da potraje)...")
    p["2D_PCA"] = p["PCA-50"][:, :2]
    t = time.time()
    p["2D_tSNE"] = TSNE(n_components=2, perplexity=30, init="pca",
                        random_state=seed).fit_transform(p["PCA-50"])
    info["vreme_tSNE_s"] = time.time() - t
    p["2D_UMAP"] = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                             random_state=seed).fit_transform(X)
    p["2D_AE"] = PCA(n_components=2).fit_transform(p["AE-10"])

    np.savez(kes, n=len(X), epohe=epohe, susedi=susedi, info=info, **p)
    return p, info


def prikaz_2d(p, y, nazivi, d):
    print("\n[1f] Poređenje 2D prikaza (boje = prave oznake, samo za tumačenje)")
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for ax, (k, naslov) in zip(axes.flat, [("2D_PCA", "PCA"), ("2D_tSNE", "t-SNE"),
                                          ("2D_UMAP", "UMAP"), ("2D_AE", "Autoenkoder (latent -> PCA 2D)")]):
        rasipanje(ax, p[k], y, naslov, nazivi)
    sacuvaj(fig, d / "1f_2d_poredjenje.png")


def osetljivost(X, y, d, brzo, seed=0):
    print("\n[1g] Osetljivost t-SNE i UMAP na parametre")
    m = min(len(X), 1500 if brzo else 3000)
    idx = np.random.default_rng(seed).choice(len(X), m, replace=False)
    Xs, ys = X[idx], y[idx]
    Xs_pca = PCA(n_components=min(50, X.shape[1])).fit_transform(Xs)

    perp = [5, 30, 100]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, pr in zip(axes, perp):
        Z = TSNE(perplexity=min(pr, (m - 1) / 3), init="pca", random_state=seed).fit_transform(Xs_pca)
        rasipanje(ax, Z, ys, f"t-SNE, perplexity={pr}", s=3)
    sacuvaj(fig, d / "1g_tsne_perplexity.png")

    susedi, razmaci = [5, 15, 50], [0.0, 0.1, 0.5]
    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    for i, nn_ in enumerate(susedi):
        for j, md in enumerate(razmaci):
            Z = umap.UMAP(n_neighbors=nn_, min_dist=md, random_state=seed).fit_transform(Xs)
            rasipanje(axes[i, j], Z, ys, f"UMAP n_neighbors={nn_}, min_dist={md}", s=3)
    sacuvaj(fig, d / "1h_umap_parametri.png")


def pokreni(X, y, nazivi, vel, d, args):
    print("\n" + "=" * 60 + "\nDEO 1: SMANJENJE DIMENZIONALNOSTI\n" + "=" * 60)
    info = pca_analiza(X, vel, d)
    p, info2 = izracunaj_prostore(X, vel, d, args.epohe, args.umap_susedi)
    info.update(info2)
    prikaz_2d(p, y, nazivi, d)
    if not args.bez_osetljivosti:
        osetljivost(X, y, d, args.brzo)
    return p, info
