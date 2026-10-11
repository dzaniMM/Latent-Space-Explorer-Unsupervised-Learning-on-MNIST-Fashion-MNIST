"""DEO 3: Analiza rezultata - koje klase se mešaju i da li su zamene stvarna
vizuelna sličnost ili posledica uparivanja klastera sa klasama."""
import numpy as np
from sklearn.metrics.cluster import contingency_matrix

from grafici import matrica_klastera, mreza_slika, sacuvaj, plt
from metrike import klasteri_u_klase


def struktura_klastera(y, kl, prag=0.2):
    """Koje klase su podeljene na više klastera i koji klasteri spajaju više
    klasa. Bez ovoga se zamena tipa '1 -> 9' lako pogrešno protumači: ako je
    klasa 1 podeljena na dva klastera, a 4 i 9 spojene u jedan, Mađarski
    algoritam višak klaster jedinica mora da upari sa klasom 9."""
    klase, klasteri = np.unique(y), np.unique(kl)
    cm = contingency_matrix(y, kl)
    dominantna = {k: klase[cm[:, j].argmax()] for j, k in enumerate(klasteri)}
    vrednosti = list(dominantna.values())
    podeljene = [c for c in klase if vrednosti.count(c) >= 2]
    udeo = cm / cm.sum(axis=0, keepdims=True)
    spojeni = {k: list(klase[udeo[:, j] >= prag]) for j, k in enumerate(klasteri)
               if (udeo[:, j] >= prag).sum() >= 2}
    return dominantna, podeljene, spojeni


def pokreni(X, y, nazivi, vel, df, oznake, d):
    print("\n" + "=" * 60 + "\nDEO 3: ANALIZA\n" + "=" * 60)
    info = {}

    # 1) Najbolja kombinacija i matrica klasteri x klase
    naj = df.sort_values("tacnost", ascending=False).iloc[0]
    kljuc = f"{naj.prostor}|{naj.algoritam}"
    kl = oznake[kljuc]
    info["najbolja_kombinacija"] = kljuc
    info["najbolja_tacnost"] = float(naj.tacnost)
    print(f"  Najbolja kombinacija: {naj.algoritam} na {naj.prostor} (tačnost {naj.tacnost:.1%})")
    matrica_klastera(y, kl, nazivi, f"{naj.algoritam} na {naj.prostor}: klase vs klasteri",
                     d / "3a_matrica_klasteri.png")

    # 2) Koje klase se najčešće mešaju
    pred = klasteri_u_klase(y, kl)
    klase = np.unique(y)
    M = np.zeros((len(klase), len(klase)))
    for i, a in enumerate(klase):
        for j, b in enumerate(klase):
            M[i, j] = np.sum((y == a) & (pred == b))
    M_norm = M / M.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(M_norm, cmap="Reds", vmin=0, vmax=1)
    for i in range(len(klase)):
        for j in range(len(klase)):
            ax.text(j, i, f"{M_norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M_norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(klase)), [nazivi[k] for k in klase], rotation=45, ha="right")
    ax.set_yticks(range(len(klase)), [nazivi[k] for k in klase])
    ax.set_xlabel("Klasa dodeljena klasteru"); ax.set_ylabel("Prava klasa")
    ax.set_title("Udeo slika svake klase po dodeljenoj klasi")
    sacuvaj(fig, d / "3b_mesanje_klasa.png")

    dominantna, podeljene, spojeni = struktura_klastera(y, kl)
    info["podeljene_klase"] = [nazivi[c] for c in podeljene]
    info["spojeni_klasteri"] = {f"K{k}": [nazivi[c] for c in v] for k, v in spojeni.items()}
    print(f"  Klase podeljene na više klastera: {info['podeljene_klase'] or 'nema'}")
    print(f"  Klasteri koji spajaju više klasa (>=20% klastera): "
          f"{info['spojeni_klasteri'] or 'nema'}")

    van = M_norm.copy(); np.fill_diagonal(van, 0)
    parovi = sorted(((van[i, j], i, j) for i in range(len(klase)) for j in range(len(klase))
                     if van[i, j] > 0), reverse=True)[:6]
    print("  Najčešće zamene (prava -> dodeljena):")
    info["najcesce_zamene"] = []
    for v, i, j in parovi:
        # Ako su pogrešno dodeljene slike uglavnom u klasteru u kome je njihova
        # klasa većinska, nisu "pomešane", već je klasa podeljena pa je višak
        # klaster pri uparivanju dobio tuđu oznaku.
        u_kl = kl[(y == klase[i]) & (pred == klase[j])]
        artefakt = len(u_kl) > 0 and dominantna[np.bincount(u_kl).argmax()] == klase[i]
        vrsta = ("artefakt uparivanja: klasa podeljena na više klastera" if artefakt
                 else "stvarno mešanje (iste slike u zajedničkom klasteru)")
        opis = f"{nazivi[klase[i]]} -> {nazivi[klase[j]]}: {v:.1%} [{vrsta}]"
        print(f"    {opis}")
        info["najcesce_zamene"].append(opis)

    # Primeri pogrešno grupisanih slika za 3 najčešće zamene
    slike, naslovi = [], []
    for _, i, j in parovi[:3]:
        idx = np.where((y == klase[i]) & (pred == klase[j]))[0][:10]
        slike += list(X[idx]); naslovi += [f"{nazivi[klase[i]]}→{nazivi[klase[j]]}"] * len(idx)
    if slike:
        fig = mreza_slika(np.array(slike), vel, naslovi, kolone=10,
                          naslov="Primeri pogrešno grupisanih slika (prava → dodeljena)")
        sacuvaj(fig, d / "3c_primeri_gresaka.png")

    return info
