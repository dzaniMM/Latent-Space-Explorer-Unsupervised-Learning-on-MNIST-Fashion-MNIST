"""Mere kvaliteta klasterovanja. Oznake se koriste SAMO ovde, za evaluaciju."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix


def tacnost(y, kl):
    """Klasteri nemaju imena, pa ih Mađarskim algoritmom uparujemo 1-na-1 sa
    klasama tako da poklapanje bude maksimalno. Višak klastera računa se kao greška."""
    cm = contingency_matrix(y, kl)
    r, c = linear_sum_assignment(-cm)
    return cm[r, c].sum() / len(y)


def klasteri_u_klase(y, kl):
    """Svakom klasteru dodeljuje klasu (Mađarski algoritam, a višak klastera
    dobija najčešću klasu). Koristi se za analizu grešaka."""
    klase, klasteri = np.unique(y), np.unique(kl)
    cm = contingency_matrix(y, kl)
    mapa = {}
    r, c = linear_sum_assignment(-cm)
    for ri, ci in zip(r, c):
        mapa[klasteri[ci]] = klase[ri]
    for j, k in enumerate(klasteri):
        mapa.setdefault(k, klase[cm[:, j].argmax()])
    return np.array([mapa[k] for k in kl])


def sve_mere(y, kl):
    return {
        "tacnost": tacnost(y, kl),
        "ARI": adjusted_rand_score(y, kl),
        "br_klastera": len(np.unique(kl)),
    }
