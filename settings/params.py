"""Settings — Système de Recommandation de Films"""
from pathlib import Path

# ── Répertoires ───────────────────────────────────────────────────────────────
HOME_DIR = Path(__file__).resolve().parent.parent

# Architecture data standard MLOps : raw → interim → processed
DATA_DIR       = HOME_DIR / "data"
DATA_DIR_RAW   = DATA_DIR / "raw"        # données brutes téléchargées (immuables)
DATA_DIR_INTERIM   = DATA_DIR / "interim"    # données nettoyées / filtrées
DATA_DIR_PROCESSED = DATA_DIR / "processed"  # features prêtes pour la modélisation

MODEL_DIR    = HOME_DIR / "models"
METRICS_DIR  = HOME_DIR / "metrics"
REPORT_DIR   = HOME_DIR / "reports"
METRICS_FILE = METRICS_DIR / "train.json"
SRC_DIR     = HOME_DIR / "src"
NOTEBOOK_DIR = HOME_DIR / "notebooks"

for _d in [DATA_DIR_RAW, DATA_DIR_INTERIM, DATA_DIR_PROCESSED, MODEL_DIR, METRICS_DIR, REPORT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Dataset MovieLens ─────────────────────────────────────────────────────────
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_ZIP = DATA_DIR_RAW / "ml-latest-small.zip"
MOVIELENS_DIR = DATA_DIR_RAW / "ml-latest-small"   # dossier extrait (brut)

# Fichiers bruts (raw)
RATINGS_FILE = MOVIELENS_DIR / "ratings.csv"
MOVIES_FILE  = MOVIELENS_DIR / "movies.csv"
TAGS_FILE    = MOVIELENS_DIR / "tags.csv"
LINKS_FILE   = MOVIELENS_DIR / "links.csv"

# Fichiers intermédiaires (interim) — nettoyés, filtrés
RATINGS_INTERIM = DATA_DIR_INTERIM / "ratings_filtered.csv"
MOVIES_INTERIM  = DATA_DIR_INTERIM / "movies_clean.csv"
TAGS_INTERIM    = DATA_DIR_INTERIM / "tags_clean.csv"

# Fichiers finaux (processed) — features prêtes pour la modélisation
USER_ITEM_MATRIX_FILE = DATA_DIR_PROCESSED / "user_item_matrix.csv"
USER_FEATURES_FILE    = DATA_DIR_PROCESSED / "user_features.csv"
ITEM_FEATURES_FILE    = DATA_DIR_PROCESSED / "item_features.csv"
GENRE_MATRIX_FILE     = DATA_DIR_PROCESSED / "genre_matrix.csv"

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = f"sqlite:///{HOME_DIR / 'mlflow.db'}"
EXPERIMENT_NAME     = "movie_recommendation"

# ── Modélisation ──────────────────────────────────────────────────────────────
SEED     = 42
TIMEZONE = "UTC"

MODEL_PARAMS = {
    "TEST_SIZE"  : 0.20,
    "N_SPLITS"   : 5,
    "TOP_N"      : 10,
    "MIN_RATINGS": 5,
    "SCALE"      : (0.5, 5.0),

    # SVD — modèle principal (Matrix Factorization)
    "SVD": {
        "n_factors"   : 100,
        "n_epochs"    : 20,
        "lr_all"      : 0.005,
        "reg_all"     : 0.02,
        "random_state": SEED,
    },

    # KNN User-Based
    "KNN_USER": {
        "k"          : 40,
        "sim_options": {"name": "cosine", "user_based": True},
    },

    # KNN Item-Based
    "KNN_ITEM": {
        "k"          : 40,
        "sim_options": {"name": "cosine", "user_based": False},
    },

    # Optuna — plages de recherche pour SVD
    "OPTUNA": {
        "n_trials"        : 30,
        "n_factors_range" : (50, 200),
        "n_epochs_range"  : (10, 50),
        "lr_range"        : (1e-4, 0.1),
        "reg_range"       : (1e-4, 0.1),
    },
}

MODEL_NAME = "model_recommender.dill"
