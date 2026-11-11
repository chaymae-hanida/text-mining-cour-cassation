# -*- coding: utf-8 -*-
"""
Étape 2 du NER : entraînement d'un CRF + validation croisée (5 plis, groupés par document).
Les plis sont groupés par document pour éviter qu'une même décision soit à la fois
dans l'entraînement et dans le test.
Sortie : models/ner_crf.joblib, metrics/ner_metrics.json
"""
import os, json, joblib
import numpy as np
import sklearn_crfsuite
from sklearn.model_selection import GroupKFold
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from preannotation_bio import annoter_ligne, ENTITES


def lire_corpus(fichier_liste="annotations/fichiers_choisis.txt"):
    """Retourne X_phrases (liste de listes de tokens), y (étiquettes), groupes (id document)."""
    phrases, labels, groupes = [], [], []
    for gid, chemin in enumerate(open(fichier_liste, encoding="utf-8").read().split()):
        texte = open(chemin, encoding="utf-8", errors="ignore").read()
        for ligne in texte.splitlines():
            if not ligne.strip():
                continue
            paires = annoter_ligne(ligne)
            phrases.append([t for t, _ in paires])
            labels.append([l for _, l in paires])
            groupes.append(gid)
    return phrases, labels, groupes


def features_mot(phrase, i):
    mot = phrase[i]
    f = {
        "bias": 1.0,
        "mot": mot,
        "pref2": mot[:2], "pref3": mot[:3],
        "suf2": mot[-2:], "suf3": mot[-3:],
        "est_chiffre": mot.isdigit(),
        "contient_chiffre": any(c.isdigit() for c in mot),
        "contient_slash": "/" in mot,
        "longueur": min(len(mot), 12),
        "forme": "".join("9" if c.isdigit() else ("/" if c == "/" else "a") for c in mot)[:8],
        "debut_phrase": i == 0,
        "fin_phrase": i == len(phrase) - 1,
    }
    for d in (-2, -1, 1, 2):
        j = i + d
        if 0 <= j < len(phrase):
            f[f"{d}:mot"] = phrase[j]
            f[f"{d}:contient_chiffre"] = any(c.isdigit() for c in phrase[j])
        else:
            f[f"{d}:bord"] = True
    return f


def phrase_features(phrase):
    return [features_mot(phrase, i) for i in range(len(phrase))]


def nouveau_crf():
    return sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1,
                                max_iterations=100, all_possible_transitions=True)


def main():
    os.makedirs("models", exist_ok=True); os.makedirs("metrics", exist_ok=True)
    phrases, labels, groupes = lire_corpus()
    X = [phrase_features(p) for p in phrases]
    groupes = np.array(groupes)
    print(f"{len(phrases)} phrases, {len(set(groupes.tolist()))} documents")

    # ---- validation croisée (5 plis groupés par document)
    plis = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, labels, groupes), 1):
        crf = nouveau_crf()
        crf.fit([X[i] for i in tr], [labels[i] for i in tr])
        pred = crf.predict([X[i] for i in te])
        vrai = [labels[i] for i in te]
        res = {"pli": k,
               "precision": precision_score(vrai, pred),
               "rappel": recall_score(vrai, pred),
               "f1": f1_score(vrai, pred)}
        plis.append(res)
        print(f"pli {k}: P={res['precision']:.3f}  R={res['rappel']:.3f}  F1={res['f1']:.3f}")

    resume = {m: (float(np.mean([p[m] for p in plis])), float(np.std([p[m] for p in plis])))
              for m in ("precision", "rappel", "f1")}
    print("Moyenne (± écart-type) :", {k: f"{v[0]:.3f} ± {v[1]:.3f}" for k, v in resume.items()})

    # ---- rapport par entité (une passe : 80 % documents train / 20 % test)
    tr, te = next(GroupKFold(n_splits=5).split(X, labels, groupes))
    crf = nouveau_crf(); crf.fit([X[i] for i in tr], [labels[i] for i in tr])
    rapport = classification_report([labels[i] for i in te], crf.predict([X[i] for i in te]),
                                    output_dict=True, zero_division=0)
    par_entite = {e: {k: float(v) for k, v in rapport[e].items()} for e in ENTITES if e in rapport}
    print(classification_report([labels[i] for i in te], crf.predict([X[i] for i in te]), zero_division=0))

    # ---- modèle final sur tout le corpus
    final = nouveau_crf(); final.fit(X, labels)
    joblib.dump(final, "models/ner_crf.joblib")
    json.dump({"cross_validation": plis, "moyenne_ecart_type": resume, "par_entite": par_entite},
              open("metrics/ner_metrics.json", "w"), ensure_ascii=False, indent=2)
    print("Modèle enregistré : models/ner_crf.joblib")


if __name__ == "__main__":
    main()
