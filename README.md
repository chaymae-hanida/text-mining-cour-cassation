# Text Mining – Décisions de la Cour de cassation (Maroc)

Projet du module Text Mining (Master MLAIM, USMBA Fès) sur deux chambres :
chambre immobilière et chambre du statut personnel et des successions.

## Fonctionnalités
- **NER** (CRF, étiquetage BIO) : DATE, NUM_DECISION, NUM_DOSSIER, ARTICLE, CODE_LOI
- **Classification** supervisée (Naïve Bayes)
- **Recherche** TF-IDF + similarité cosinus
- **RAG** avec choix du LLM (ChatGPT / DeepSeek)
- **Interface web** Flask et **métriques** (validation croisée à 5 plis)

## Résultats (validation croisée 5 plis)
| Module | Résultat |
|---|---|
| NER (CRF) | F1 = 0,968 |
| Classification (Naïve Bayes) | accuracy = 0,996, F1 macro = 0,993 |

Limite : les étiquettes du NER viennent d'une pré-annotation par règles (regex),
non corrigée entièrement à la main.

## Installation
```
pip install flask scikit-learn sklearn-crfsuite seqeval joblib requests
```
Place les fichiers `.txt` dans `data/immobilier/` et `data/statut_personnel/`.

## Lancer
```
python preannotation_bio.py
python train_ner.py
python train_classif_search.py
python app.py
```
Puis ouvrir http://127.0.0.1:5000

## RAG
Définir `OPENAI_API_KEY` ou `DEEPSEEK_API_KEY` (ou coller la clé dans l'interface).
Ne jamais publier une clé API sur GitHub.
