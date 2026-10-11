"""
Projekat: Nenadgledano mašinsko učenje na MNIST i Fashion-MNIST
================================================================
Pokretanje (primeri):
    python glavni.py --dataset oba                 # ceo projekat za oba skupa
    python glavni.py --dataset mnist --delovi 1 2  # samo delovi 1 i 2
    python glavni.py --dataset digits --brzo       # brza proba bez interneta

Rezultati (grafici, CSV tabele, sazetak.txt) idu u folder rezultati/<skup>/.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import deo1_dimenzionalnost as deo1
import deo2_klasterovanje as deo2
import deo3_analiza as deo3
from podaci import ucitaj_sve, uzorak


def jedan_skup(ime, args):
    t0 = time.time()
    d = Path(args.izlaz) / ime
    d.mkdir(parents=True, exist_ok=True)
    X_sve, y_sve, nazivi, vel = ucitaj_sve(ime)
    # mali skup (digits) ima malo koraka po epohi, pa mu treba više epoha
    args.epohe = args.epohe_iz_cli or (150 if ime == "digits" else 20)
    if args.brzo:
        args.epohe = max(5, args.epohe // 4)
    idx = uzorak(len(X_sve), args.n, seed=42)
    X, y = X_sve[idx], y_sve[idx]
    print(f"\n##### Skup: {ime} | ukupno {X_sve.shape}, uzorak za delove 1, 2 i 3: {X.shape} #####")

    # Kada se pokreće samo deo projekta, dopunjuje se postojeći sažetak umesto
    # da se izgube brojevi iz delova koji ovaj put nisu pokrenuti
    stari = d / "sazetak.txt"
    sazetak = json.loads(stari.read_text(encoding="utf-8")) if stari.exists() else {}
    if sazetak.get("uzorak") != len(X):
        sazetak = {}
    sazetak.update({"skup": ime, "uzorak": len(X)})
    p = df2 = oznake = None

    # deo 3 koristi klastere iz dela 2; ako još ne postoje, deo 2 se pokreće automatski
    treba_deo2 = 2 in args.delovi or (3 in args.delovi and not (d / "2_klasteri.npz").exists())
    if 1 in args.delovi or treba_deo2:
        if 1 in args.delovi:
            p, info = deo1.pokreni(X, y, nazivi, vel, d, args)
        else:  # delu 2 trebaju reprezentacije iz dela 1
            p, info = deo1.izracunaj_prostore(X, vel, d, args.epohe, args.umap_susedi)
        sazetak.setdefault("deo1", {}).update(info)  # bez dela 1 info nema PCA pragove

    if treba_deo2:
        df2, oznake = deo2.pokreni(X, y, p, d, args)
    elif 3 in args.delovi:
        df2 = pd.read_csv(d / "2_rezultati_klasterovanja.csv")
        oznake = dict(np.load(d / "2_klasteri.npz"))
    if df2 is not None:
        naj = df2.sort_values("tacnost", ascending=False).iloc[0]
        sazetak["deo2_najbolje"] = f"{naj.algoritam} na {naj.prostor}: tačnost {naj.tacnost:.3f}, ARI {naj.ARI:.3f}"
        sazetak["deo2_kmeans_pikseli_tacnost"] = float(
            df2[df2.prostor == "sirovi pikseli"].tacnost.iloc[0])

    if 3 in args.delovi:
        sazetak["deo3"] = deo3.pokreni(X, y, nazivi, vel, df2, oznake, d)

    sazetak["ukupno_vreme_min"] = round((time.time() - t0) / 60, 1)
    with open(d / "sazetak.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(sazetak, ensure_ascii=False, indent=2, default=float))
    print(f"\nGotovo za {sazetak['ukupno_vreme_min']} min. Sažetak: {d / 'sazetak.txt'}")
    return sazetak


def poredjenje(sazeci, izlaz):
    """Tabela MNIST vs Fashion-MNIST, kada se pokrene --dataset oba."""
    redovi = []
    for ime in sazeci:
        d = Path(izlaz) / ime
        if (d / "2_rezultati_klasterovanja.csv").exists():
            t = pd.read_csv(d / "2_rezultati_klasterovanja.csv")
            t["skup"] = ime
            redovi.append(t)
    if len(redovi) == 2:
        t = pd.concat(redovi)
        tab = t.pivot_table(index=["prostor", "algoritam"], columns="skup", values=["tacnost", "ARI"])
        tab.to_csv(Path(izlaz) / "poredjenje_mnist_fashion.csv", float_format="%.3f")
        print("\nPoređenje skupova (tačnost):")
        print(tab["tacnost"].round(3).to_string())


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--dataset", choices=["mnist", "fashion", "digits", "oba"], default="oba")
    pa.add_argument("--n", type=int, default=70000,
                    help="veličina uzorka (podrazumevano 70000 = ceo skup)")
    pa.add_argument("--delovi", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    pa.add_argument("--epohe", type=int, default=None,
                    help="broj epoha autoenkodera (podrazumevano 20, za digits 150)")
    pa.add_argument("--umap_susedi", type=int, default=30,
                    help="n_neighbors za UMAP-10 i AE-UMAP-10 (podrazumevano 30)")
    pa.add_argument("--brzo", action="store_true", help="manje računanja (za probu)")
    pa.add_argument("--bez_osetljivosti", action="store_true",
                    help="preskoči analizu osetljivosti t-SNE/UMAP (najsporiji deo)")
    pa.add_argument("--izlaz", default="rezultati")
    args = pa.parse_args()
    args.epohe_iz_cli = args.epohe

    skupovi = ["mnist", "fashion"] if args.dataset == "oba" else [args.dataset]
    sazeci = {ime: jedan_skup(ime, args) for ime in skupovi}
    if args.dataset == "oba":
        poredjenje(sazeci, args.izlaz)


if __name__ == "__main__":
    main()
