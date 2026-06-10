# Système de Recommandation de Films — MLOps

**Auteur** : Moussa Seye — EPT DIC3  
**Encadrante** : Mme Mously Diaw  
**Dataset** : [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/) — 100 836 notes, 610 utilisateurs, 9 742 films  
**Tracking** : MLflow (backend SQLite)

---

## Architecture du projet

```
projet-moussa-seye/
├── data/
│   ├── raw/                ← données brutes téléchargées (gitignored)
│   ├── interim/            ← données nettoyées / filtrées en Parquet (gitignored)
│   └── processed/          ← features prêtes pour la modélisation (gitignored)
├── models/                 ← modèles sérialisés .dill (gitignored)
├── notebooks/
│   └── 01-eda.ipynb        ← pipeline complet + analyse exploratoire
├── reports/                ← graphiques générés (gitignored)
├── src/
│   ├── data_loader.py      ← téléchargement + nettoyage (raw → interim)
│   ├── features.py         ← feature engineering (interim → processed)
│   ├── trainer.py          ← entraînement 4 modèles + Optuna + MLflow
│   └── recommender.py      ← génération Top-N recommandations
├── settings/
│   └── params.py           ← configuration centralisée (chemins, hyperparamètres)
├── tests/
│   └── test_pipeline.py    ← 25 tests unitaires pytest
├── requirements.txt
└── .gitignore
```

**Pipeline de données** :
```
download_movielens()  →  data/raw/ml-latest-small/
process_ratings()     →  data/interim/ratings_filtered.parquet   (filtre MIN_RATINGS=5)
process_movies()      →  data/interim/movies_clean.parquet       (extraction année, genres)
build_user_features() →  data/processed/user_features.parquet
build_item_features() →  data/processed/item_features.parquet
build_user_item_matrix() → data/processed/user_item_matrix.parquet
```

---

## Installation

### 1. Créer et activer l'environnement conda

```bash
conda create -n mlops-reco python=3.10 -y
conda activate mlops-reco
```

### 2. Installer les dépendances

`scikit-surprise` nécessite numpy et Cython au moment du build — installer dans cet ordre :

```bash
pip install numpy==1.26.4 cython
pip install scikit-surprise==1.1.3 --no-build-isolation
pip install "setuptools<70"
pip install -r requirements.txt
```

---

## Lancer le projet

### Option A — Via le notebook (recommandé pour visualiser le pipeline)

```bash
jupyter notebook notebooks/01-eda.ipynb
```

Le notebook couvre de bout en bout :
1. Téléchargement du dataset (`data/raw/`)
2. Nettoyage et filtrage (`data/interim/`)
3. Feature engineering (`data/processed/`)
4. EDA complète avec visualisations
5. Log des métriques dans MLflow

### Option B — Via les scripts Python (CLI)

```bash
# Étape 1 : télécharger le dataset et construire interim/
python src/data_loader.py

# Étape 2 : construire les features (processed/)
python src/features.py

# Étape 3 : entraîner les modèles (avec Optuna pour SVD)
python src/trainer.py

# Générer des recommandations pour un utilisateur
python src/recommender.py
```

### Visualiser les expériences MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Ouvrir [http://localhost:5000](http://localhost:5000)

### Lancer les tests

```bash
pytest tests/ -v
```

---

## Modèles

| Modèle | Description |
|---|---|
| `NormalPredictor` | Baseline — prédiction aléatoire selon la distribution des notes |
| `KNNBasic` (user-based) | Collaborative filtering — similarité cosinus entre utilisateurs |
| `KNNBasic` (item-based) | Collaborative filtering — similarité cosinus entre films |
| `SVD` | Matrix Factorization — modèle principal |
| `SVD + Optuna` | SVD avec optimisation automatique des hyperparamètres (30 trials) |

---

## Pratiques MLOps appliquées

| Pratique | Outil / Méthode |
|---|---|
| Tracking des expériences | MLflow (backend SQLite) |
| Versioning des paramètres et métriques | MLflow Params + Metrics |
| Optimisation des hyperparamètres | Optuna + MLflowCallback |
| Architecture data | `raw/` → `interim/` → `processed/` |
| Logging structuré | Loguru (format UTC) |
| Sérialisation des modèles | Dill (préfixe `YYYYMMDD_`) |
| Configuration centralisée | `settings/params.py` |
| Tests automatisés | pytest (25 tests) |
| Reproductibilité | `SEED = 42`, `requirements.txt` |
