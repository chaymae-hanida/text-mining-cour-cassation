# Projet Text Mining – Décisions de la Cour de cassation

## Installation
pip install flask scikit-learn sklearn-crfsuite seqeval joblib requests

## Organisation des données (à mettre à côté des scripts)
data/immobilier/*.txt            (contenu de الغرفة_العقارية.rar)
data/statut_personnel/*.txt      (contenu de غرفة_الأحوال_الشخصية_و_الميراث.zip)

## Ordre d'exécution
1. python preannotation_bio.py        -> annotations/pre_annotation.conll (100 décisions, 5 entités)
2. (corriger les étiquettes à la main si possible)
3. python train_ner.py                -> models/ner_crf.joblib + metrics/ner_metrics.json
4. python train_classif_search.py     -> models/clf_nb.joblib + models/tfidf_index.joblib + metrics/clf_metrics.json
5. python app.py                      -> http://127.0.0.1:5000

## RAG (ChatGPT / DeepSeek)
Définir OPENAI_API_KEY ou DEEPSEEK_API_KEY (ou coller la clé dans l'interface).
