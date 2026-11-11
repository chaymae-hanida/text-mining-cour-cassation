# -*- coding: utf-8 -*-
"""
Classification (Naïve Bayes) + index de recherche (TF-IDF / cosinus) sur les 2 chambres.
- On retire l'en-tête (5 premières lignes) et les mots qui donnent directement la classe
  (عقاري / شرعي ...) pour éviter la fuite d'étiquette.
- Validation croisée stratifiée à 5 plis.
Sorties : models/clf_nb.joblib, models/tfidf_index.joblib, metrics/clf_metrics.json
"""
import os, re, glob, json, joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from outils_texte import normaliser, prep_recherche

CLASSES = {"data/immobilier": "immobilier", "data/statut_personnel": "statut_personnel"}
FUITE = re.compile(r"عقاري\w*|الشرعي|شرعي\w*|التحفيظ|تحفيظ")
def nettoyer_pour_classif(texte):
    lignes = texte.splitlines()[5:]          # on enlève l'en-tête
    t = normaliser(" ".join(lignes))
    t = re.sub(r"\d+", " ", t)
    return FUITE.sub(" ", t)


def charger():
    docs, y, noms = [], [], []
    for dossier, classe in CLASSES.items():
        for f in sorted(glob.glob(f"{dossier}/*.txt")):
            t = open(f, encoding="utf-8", errors="ignore").read()
            if len(t.strip()) < 200:
                continue
            docs.append(t); y.append(classe); noms.append(f"{classe}/{os.path.basename(f)}")
    return docs, np.array(y), noms


def main():
    os.makedirs("models", exist_ok=True); os.makedirs("metrics", exist_ok=True)
    docs, y, noms = charger()
    print(f"{len(docs)} documents :", {c: int((y == c).sum()) for c in set(y)})

    # ---------- classification
    X = [nettoyer_pour_classif(d) for d in docs]
    pipe = make_pipeline(TfidfVectorizer(min_df=3, max_df=0.8, ngram_range=(1, 2), sublinear_tf=True),
                         MultinomialNB(alpha=0.1))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"])
    resume = {k.replace("test_", ""): [float(np.mean(v)), float(np.std(v))]
              for k, v in scores.items() if k.startswith("test_")}
    pred = cross_val_predict(pipe, X, y, cv=cv)
    print({k: f"{v[0]:.3f} ± {v[1]:.3f}" for k, v in resume.items()})
    print(classification_report(y, pred, digits=3))
    labels = sorted(set(y))
    cm = confusion_matrix(y, pred, labels=labels).tolist()
    pipe.fit(X, y)
    joblib.dump(pipe, "models/clf_nb.joblib")
    json.dump({"cross_validation": resume, "classes": labels, "matrice_confusion": cm},
              open("metrics/clf_metrics.json", "w"), ensure_ascii=False, indent=2)

    # ---------- index de recherche (TF-IDF sur le texte complet)
    vec = TfidfVectorizer(preprocessor=prep_recherche,
                          min_df=2, max_df=0.9, sublinear_tf=True)
    M = vec.fit_transform(docs)
    joblib.dump({"vec": vec, "matrice": M, "noms": noms, "docs": docs, "classes": y.tolist()},
                "models/tfidf_index.joblib")
    print("Index TF-IDF :", M.shape)


if __name__ == "__main__":
    main()
