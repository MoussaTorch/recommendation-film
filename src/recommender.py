"""
recommender.py
--------------
Interface de recommandation : génère le Top-N pour un utilisateur donné
à partir d'un modèle entraîné (SVD ou KNN).
"""
import sys
from pathlib import Path
from collections import defaultdict

import dill
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings.params import MODEL_DIR, MODEL_PARAMS

log_fmt = (
    "<green>{time:YYYY-MM-DD HH:mm:ss!UTC}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}"
)
logger.remove()
logger.add(sys.stderr, format=log_fmt, level="INFO")


# ── Chargement du modèle ──────────────────────────────────────────────────────
def load_model(model_path: Path):
    """Charge un modèle sérialisé avec dill."""
    with open(model_path, "rb") as f:
        model = dill.load(f)
    logger.info(f"Modèle chargé : {model_path.name}")
    return model


def get_latest_model(pattern: str = "*.dill") -> Path:
    """Retourne le modèle le plus récent dans MODEL_DIR."""
    models = sorted(MODEL_DIR.glob(pattern))
    if not models:
        raise FileNotFoundError(f"Aucun modèle trouvé dans {MODEL_DIR}")
    return models[-1]


# ── Top-N Recommendations ─────────────────────────────────────────────────────
def get_top_n_recommendations(
    algo,
    user_id: int,
    all_movie_ids: list[int],
    rated_movie_ids: set[int],
    movies: pd.DataFrame,
    n: int = MODEL_PARAMS["TOP_N"],
) -> pd.DataFrame:
    """
    Pour un utilisateur donné, prédit la note pour tous les films
    non encore vus, trie par note prédite et retourne le Top-N.

    Paramètres
    ----------
    algo            : modèle Surprise entraîné
    user_id         : identifiant de l'utilisateur cible
    all_movie_ids   : liste de tous les movieId du dataset
    rated_movie_ids : films déjà notés par l'utilisateur
    movies          : DataFrame avec movieId, title_clean, genres
    n               : nombre de recommandations
    """
    unseen_movies = [m for m in all_movie_ids if m not in rated_movie_ids]

    # Prédiction en batch
    predictions = [algo.predict(user_id, movie_id) for movie_id in unseen_movies]
    predictions.sort(key=lambda x: x.est, reverse=True)

    top_n = predictions[:n]
    results = []
    for pred in top_n:
        movie_info = movies[movies["movieid"] == pred.iid]
        title  = movie_info["title_clean"].values[0] if len(movie_info) else f"Film {pred.iid}"
        genres = movie_info["genres"].values[0]      if len(movie_info) else "N/A"
        results.append({
            "movieid"          : pred.iid,
            "title"            : title,
            "genres"           : genres,
            "predicted_rating" : round(pred.est, 3),
        })

    df_reco = pd.DataFrame(results)
    logger.info(f"Top-{n} recommandations pour l'utilisateur {user_id} :\n{df_reco.to_string(index=False)}")
    return df_reco


# ── Recommandations en batch ──────────────────────────────────────────────────
def get_top_n_all_users(
    algo,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    n: int = MODEL_PARAMS["TOP_N"],
) -> dict[int, pd.DataFrame]:
    """
    Génère le Top-N pour tous les utilisateurs du dataset.
    Retourne un dictionnaire {user_id: DataFrame recommandations}.
    """
    all_movie_ids = ratings["movieid"].unique().tolist()
    user_rated    = ratings.groupby("userid")["movieid"].apply(set).to_dict()

    reco_dict = {}
    for user_id in ratings["userid"].unique():
        reco_dict[user_id] = get_top_n_recommendations(
            algo, user_id,
            all_movie_ids,
            user_rated.get(user_id, set()),
            movies, n,
        )

    logger.success(f"Recommandations générées pour {len(reco_dict)} utilisateurs")
    return reco_dict


if __name__ == "__main__":
    from data_loader import load_ratings, load_movies
    from features import build_surprise_dataset
    from surprise.model_selection import train_test_split
    from settings.params import SEED, MODEL_PARAMS

    ratings = load_ratings()
    movies  = load_movies()
    data    = build_surprise_dataset(ratings)

    # Charger le modèle SVD le plus récent
    model_path = get_latest_model("*model_recommender.dill")
    algo = load_model(model_path)

    # Recommandations pour l'utilisateur 1
    all_ids   = ratings["movieid"].unique().tolist()
    seen_ids  = set(ratings[ratings["userid"] == 1]["movieid"].tolist())
    reco = get_top_n_recommendations(algo, 1, all_ids, seen_ids, movies)
