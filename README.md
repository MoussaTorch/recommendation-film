# Système de Recommandation de Films — MLOps

**Auteur** : Moussa Seye — EPT DIC3  
**Dataset** : [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/)  
**Tracking** : MLflow  
**Modèles** : Baseline, KNN User-Based, KNN Item-Based, SVD

---

## Structure du projet

```
projet-moussa-seye/
├── data/
│   ├── input/          ← données brutes MovieLens (gitignored)
│   └── output/         ← données transformées
├── models/             ← modèles sérialisés (gitignored)
├── notebooks/
│   └── 01-eda.ipynb    ← analyse exploratoire complète
├── src/
│   ├── data_loader.py  ← chargement + validation des données
│   ├── features.py     ← feature engineering
│   ├── trainer.py      ← entraînement + MLflow tracking
│   └── recommender.py  ← génération des recommandations
├── settings/
│   └── params.py       ← configuration centralisée
├── tests/
│   └── test_pipeline.py← tests unitaires
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

## Télécharger les données

```bash
python src/data_loader.py
```

> Télécharge automatiquement `ml-latest-small.zip` depuis grouplens.org et l'extrait dans `data/input/`.

## Lancer les expériences

```bash
python src/trainer.py
```

## Visualiser les résultats MLflow

```bash
mlflow ui --backend-store-uri mlruns/
```
Ouvrir [http://localhost:5000](http://localhost:5000)

## Générer des recommandations

```bash
python src/recommender.py
```

---

## Pratiques MLOps appliquées

| Pratique | Outil |
|---|---|
| Tracking des expériences | MLflow |
| Versioning des paramètres | MLflow Params |
| Logging structuré | Loguru |
| Sérialisation des modèles | Dill |
| Configuration centralisée | `settings/params.py` |
| Tests automatisés | pytest |
| Reproductibilité | `SEED = 42`, `requirements.txt` |
