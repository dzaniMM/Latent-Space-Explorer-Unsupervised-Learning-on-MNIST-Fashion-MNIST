# Nenadgledano mašinsko učenje na MNIST i Fashion-MNIST

**Istraživačko pitanje:** koliko dobro različite tehnike nenadgledanog učenja otkrivaju
strukturu u slikama bez ikakvih oznaka, i zašto je Fashion-MNIST teži od MNIST-a?

Oznake se nigde ne koriste za učenje, već samo na kraju, za merenje kvaliteta.

## Instalacija i pokretanje

```
pip install -r requirements.txt
python glavni.py --dataset digits --brzo     # brza proba (~1 min, bez interneta)
python glavni.py --dataset oba               # ceo projekat za MNIST i Fashion-MNIST
```

Korisne opcije:

| Opcija | Značenje |
|---|---|
| `--dataset mnist / fashion / digits / oba` | koji skup (podrazumevano `oba`) |
| `--n 70000` | veličina uzorka (podrazumevano 70000 = ceo skup) |
| `--delovi 1 2 3` | koje delove pokrenuti: 1 = smanjenje dimenzionalnosti, 2 = klasterovanje, 3 = analiza grešaka |
| `--epohe 20` | broj epoha autoenkodera |
| `--umap_susedi 15` | `n_neighbors` za UMAP-10 i AE-UMAP-10 (podrazumevano 30) |
| `--bez_osetljivosti` | preskače najsporiji deo (mreža t-SNE/UMAP parametara) |
| `--brzo` | manje računanja, za probu |

Prvo pokretanje preuzima podatke sa OpenML-a. Reprezentacije iz dela 1 čuvaju se u
`prostori.npz`, pa ponovno pokretanje ostalih delova ide mnogo brže. Ako promeniš nešto u
delu 1, obriši taj fajl.

Sa celim skupom (n=70000) projekat traje reda veličine sat vremena po skupu; najsporiji su
t-SNE, UMAP i GMM u PCA-50 prostoru. Za bržu probu koristi npr. `--n 10000`.
Sa GPU-om se autoenkoder trenira znatno brže (PyTorch ga koristi automatski).

## Struktura

| Fajl | Sadržaj | Poglavlje izveštaja |
|---|---|---|
| `deo1_dimenzionalnost.py` | PCA (varijansa, glavne komponente, rekonstrukcije), t-SNE, UMAP, autoenkoder, osetljivost na parametre | Smanjenje dimenzionalnosti |
| `deo2_klasterovanje.py` | K-means, GMM i aglomerativno (Ward) u prostorima PCA-50, UMAP-10, AE-10, AE-UMAP-10 | Klasterovanje |
| `deo3_analiza.py` | Matrica klasteri × klase, podeljene/spojene klase, najčešće zamene (stvarne ili artefakt uparivanja) | Diskusija |
| `autoenkoder.py` | Konvolucioni AE (28×28) i MLP AE (8×8) u PyTorch-u | |
| `metrike.py` | Tačnost (Mađarski algoritam), ARI | |
| `podaci.py`, `grafici.py`, `glavni.py` | Učitavanje, crtanje, pokretanje | |

Svi rezultati idu u `rezultati/<skup>/`: grafici (numerisani po delovima), CSV tabele
i `sazetak.txt` sa ključnim brojevima. Uz `--dataset oba` nastaje i
`rezultati/poredjenje_mnist_fashion.csv`.

## Mere kvaliteta

- **Aglomerativno (Ward)** za n > 15000 spaja samo susede u grafu 15 najbližih suseda, jer bi klasičan Ward tražio desetine GB memorije.
- **Tačnost**: klasteri se Mađarskim algoritmom 1-na-1 uparuju sa klasama; višak klastera računa se kao greška.
- **ARI**: slaganje parova slika (zajedno/odvojeno) sa pravim klasama, nezavisno od imena klastera i korigovano za
  slučajnost (0 = nasumično, 1 = savršeno).

## Diskusija

- PCA vs AE rekonstrukcija sa istih 10 dimenzija (`1e`): da li nelinearnost pomaže?
- Da li UMAP na osobinama autoenkodera (AE-UMAP-10, pristup N2D) pomaže, i kako to zavisi od veličine uzorka (`--n`).
- Zašto UMAP prostor daje mnogo bolje klastere od PCA.
- Koje klase se mešaju (`3b`, `3c`) i da li je to greška modela ili stvarna vizuelna sličnost. Zamena označena kao
  "artefakt uparivanja" znači da je klasa podeljena na više klastera, pa je višak klaster dobio tuđu oznaku.
- Ograničenje: svi algoritmi dobijaju k = 10, što koristi znanje o broju klasa.
- Ograničenje: rezultati zavise od nasumičnosti (seed) i veličine uzorka; za jače zaključke ponovi sa više seed-ova.
