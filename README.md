# Système de Recommandation de Films — MLOps

**Auteur** : Moussa Seye  
**Dataset** : [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/) — 100 836 notes, 610 utilisateurs, 9 742 films  
**Tracking** : MLflow (backend SQLite)

Rapport : pour une meilleure compréhension du projet, lire `rapport_projet.md`.

---

## Architecture du projet

```
recommendation-film/
├── .github/workflows/ci.yml  ← CI/CD GitHub Actions (pytest + Docker)
├── dvc.yaml                  ← pipeline reproductible DVC
├── params.yaml               ← hyperparamètres versionnés
├── Dockerfile                ← conteneur API FastAPI
├── docker-compose.yml        ← déploiement local API + volumes
├── data/
│   ├── raw/                  ← données brutes téléchargées (gitignored)
│   ├── interim/              ← données nettoyées / filtrées CSV (gitignored)
│   └── processed/            ← features prêtes pour la modélisation (gitignored)
├── interface/                ← application web (dashboard + prédiction)
│   ├── api/main.py           ← backend FastAPI (port 8000, Swagger /docs)
│   ├── src/                  ← frontend React 18 + Vite
│   └── requirements.txt
├── metrics/
│   └── train.json            ← métriques du meilleur modèle (DVC metrics)
├── models/                   ← modèles sérialisés .dill (gitignored)
├── notebooks/
│   └── 01-eda.ipynb          ← EDA complète avec MLflow logging
├── src/
│   ├── data_loader.py        ← téléchargement + nettoyage (raw → interim)
│   ├── features.py           ← feature engineering (interim → processed)
│   ├── trainer.py            ← entraînement 4 modèles + Optuna + MLflow
│   └── recommender.py        ← génération Top-N recommandations
├── settings/
│   └── params.py             ← configuration centralisée (chemins, hyperparamètres)
├── tests/
│   ├── test_pipeline.py      ← 18 tests unitaires pytest
│   └── test_api.py           ← tests API FastAPI
├── requirements.txt
└── .gitignore
```

**Pipeline de données** :

```
download_movielens()     →  data/raw/ml-latest-small/
process_ratings()        →  data/interim/ratings_filtered.csv   (filtre MIN_RATINGS=5)
process_movies()         →  data/interim/movies_clean.csv       (extraction année, genres)
build_user_features()    →  data/processed/user_features.csv
build_item_features()    →  data/processed/item_features.csv
build_user_item_matrix() →  data/processed/user_item_matrix.csv
train models             →  models/*.dill + metrics/train.json
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

### Option A — Pipeline DVC (recommandé pour la reproductibilité)

```bash
dvc repro
```

Relance uniquement les étapes modifiées :

1. `data` — téléchargement + nettoyage (`data_loader.py`)
2. `features` — feature engineering (`features.py`)
3. `train` — entraînement 4 modèles + export métriques (`trainer.py`)

Visualiser le pipeline :

```bash
dvc dag
dvc metrics show
```

Les hyperparamètres sont versionnés dans `params.yaml` (miroir de `settings/params.py`).

### Option B — Via les scripts Python (CLI)

```bash
python src/data_loader.py
python src/features.py
python src/trainer.py                  # sans Optuna (rapide, défaut DVC)
python src/trainer.py --with-optuna      # avec Optuna (30 trials)
python src/recommender.py
```

### Option C — Via le notebook (EDA)

```bash
jupyter notebook notebooks/01-eda.ipynb
```

### Visualiser les expériences MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Ouvrir [http://localhost:5000](http://localhost:5000)

---

## API REST (FastAPI + Swagger)

### Local

```bash
uvicorn interface.api.main:app --reload --port 8000
```

- API : [http://localhost:8000/api/health](http://localhost:8000/api/health)
- Swagger : [http://localhost:8000/docs](http://localhost:8000/docs)

Endpoints principaux :


| Méthode | Route                    | Description                            |
| ------- | ------------------------ | -------------------------------------- |
| GET     | `/api/health`            | Santé de l'API                         |
| GET     | `/api/stats`             | Statistiques du dataset                |
| GET     | `/api/models`            | Comparaison des modèles                |
| POST    | `/api/recommend`         | Recommandations Top-N pour un user     |
| POST    | `/api/recommend-profile` | Recommandations depuis des films aimés |


### Docker

```bash
# Prérequis : data/interim/, data/processed/ et models/ générés (dvc repro)
docker compose up --build
```

Ou manuellement :

```bash
docker build -t cinerec-api .
docker run -p 8000:8000 \
  -v "$(pwd)/data/interim:/app/data/interim:ro" \
  -v "$(pwd)/data/processed:/app/data/processed:ro" \
  -v "$(pwd)/models:/app/models:ro" \
  cinerec-api
```

---

## Interface web (CinéRec)

**Terminal 1 — Backend** (si pas déjà lancé via Docker) :

```bash
uvicorn interface.api.main:app --reload --port 8000
```

**Terminal 2 — Frontend React** :

```bash
cd interface
npm install
npm run dev
```

Ouvrir [http://localhost:3000](http://localhost:3000)

---

## Tests & CI/CD

```bash
pytest tests/ -v
```

Le workflow GitHub Actions (`.github/workflows/ci.yml`) exécute automatiquement à chaque push :

1. Installation des dépendances
2. Pipeline data + features
3. `pytest` (pipeline + API)
4. Validation du graphe DVC (`dvc dag`)
5. Build Docker + smoke test `/api/health` et `/docs`

---

## Modèles


| Modèle                  | Description                                                     |
| ----------------------- | --------------------------------------------------------------- |
| `NormalPredictor`       | Baseline — prédiction aléatoire selon la distribution des notes |
| `KNNBasic` (user-based) | Collaborative filtering — similarité cosinus entre utilisateurs |
| `KNNBasic` (item-based) | Collaborative filtering — similarité cosinus entre films        |
| `SVD`                   | Matrix Factorization — modèle principal                         |
| `SVD + Optuna`          | SVD avec optimisation automatique (`--with-optuna`)             |


---

## Pratiques MLOps appliquées


| Pratique                     | Outil / Méthode                              |
| ---------------------------- | -------------------------------------------- |
| Tracking des expériences     | MLflow (backend SQLite, 40+ runs)            |
| Pipeline reproductible       | DVC (`dvc.yaml` + `params.yaml`)             |
| Versioning des paramètres    | `params.yaml` + `settings/params.py`         |
| Optimisation hyperparamètres | Optuna + MLflowCallback                      |
| Architecture data            | `raw/` → `interim/` → `processed/`           |
| API REST documentée          | FastAPI + Swagger auto (`/docs`)             |
| Conteneurisation             | Dockerfile + docker-compose                  |
| CI/CD                        | GitHub Actions (tests + Docker)              |
| Logging structuré            | Loguru (format UTC)                          |
| Sérialisation des modèles    | Dill (préfixe `YYYYMMDD_`)                   |
| Tests automatisés            | pytest (pipeline + API)                      |
| Reproductibilité             | `SEED = 42`, `requirements.txt`, `dvc repro` |


