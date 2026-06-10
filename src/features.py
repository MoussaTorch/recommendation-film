"""
features.py
-----------
Feature engineering : interim → processed.

Construit et persiste (CSV) :
  - Dataset Surprise       (pour l'entraînement, non persisté)
  - Matrice user-item  →   processed/user_item_matrix.csv
  - User features      →   processed/user_features.csv
  - Item features      →   processed/item_features.csv
  - Genre matrix       →   processed/genre_matrix.csv
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger
from surprise import Dataset, Reader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings.params import (
    MODEL_PARAMS,
    USER_ITEM_MATRIX_FILE,
    USER_FEATURES_FILE,
    ITEM_FEATURES_FILE,
    GENRE_MATRIX_FILE,
)

log_fmt = (
    "<green>{time:YYYY-MM-DD HH:mm:ss!UTC}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}"
)
logger.remove()
logger.add(sys.stderr, format=log_fmt, level="INFO")


# ── Dataset Surprise ──────────────────────────────────────────────────────────
def build_surprise_dataset(ratings: pd.DataFrame) -> Dataset:
    """Convertit un DataFrame ratings (userid, movieid, rating) en Dataset Surprise."""
    reader = Reader(rating_scale=MODEL_PARAMS["SCALE"])
    data = Dataset.load_from_df(ratings[["userid", "movieid", "rating"]], reader)
    logger.info(f"Dataset Surprise construit ({len(ratings):,} notes)")
    return data


# ── Matrice user-item ─────────────────────────────────────────────────────────
def build_user_item_matrix(ratings: pd.DataFrame,
                           save: bool = True) -> pd.DataFrame:
    """
    Pivot ratings → matrice user × item (sparse).
    NaN = film non noté par l'utilisateur.
    L'index (userid) est conservé dans la colonne 'userid' du CSV.
    """
    if save and USER_ITEM_MATRIX_FILE.exists():
        logger.info(f"Matrice user-item déjà présente : {USER_ITEM_MATRIX_FILE}")
        df = pd.read_csv(USER_ITEM_MATRIX_FILE, index_col=0)
        df.columns = df.columns.astype(int)  # movieId en int
        return df

    matrix = ratings.pivot_table(index="userid", columns="movieid", values="rating")
    sparsity = 1 - matrix.notna().sum().sum() / matrix.size
    logger.info(f"Matrice user-item : {matrix.shape} — sparsité {sparsity:.2%}")

    if save:
        matrix.to_csv(USER_ITEM_MATRIX_FILE)
        logger.success(f"Sauvegardé : {USER_ITEM_MATRIX_FILE}")
    return matrix


# ── Features utilisateurs ────────────────────────────────────────────────────
def build_user_features(ratings: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Statistiques agrégées par utilisateur."""
    if save and USER_FEATURES_FILE.exists():
        logger.info(f"User features déjà présentes : {USER_FEATURES_FILE}")
        return pd.read_csv(USER_FEATURES_FILE)

    uf = (
        ratings.groupby("userid")["rating"]
        .agg(
            user_n_ratings="count",
            user_mean_rating="mean",
            user_std_rating="std",
            user_min_rating="min",
            user_max_rating="max",
        )
        .round(4)
        .reset_index()
    )
    logger.info(f"User features : {uf.shape}")

    if save:
        uf.to_csv(USER_FEATURES_FILE, index=False)
        logger.success(f"Sauvegardé : {USER_FEATURES_FILE}")
    return uf


# ── Features films ────────────────────────────────────────────────────────────
def build_item_features(ratings: pd.DataFrame, movies: pd.DataFrame,
                        save: bool = True) -> pd.DataFrame:
    """Statistiques par film + métadonnées (titre, année, genres)."""
    if save and ITEM_FEATURES_FILE.exists():
        logger.info(f"Item features déjà présentes : {ITEM_FEATURES_FILE}")
        df = pd.read_csv(ITEM_FEATURES_FILE)
        df["genres_list"] = df["genres_list"].str.split("|")
        return df

    itf = (
        ratings.groupby("movieid")["rating"]
        .agg(
            item_n_ratings="count",
            item_mean_rating="mean",
            item_std_rating="std",
        )
        .round(4)
        .reset_index()
    )
    itf = itf.merge(
        movies[["movieid", "title_clean", "year", "genres", "genres_list"]],
        on="movieid",
        how="left",
    )
    logger.info(f"Item features : {itf.shape}")

    if save:
        itf_save = itf.copy()
        itf_save["genres_list"] = itf_save["genres_list"].apply(
            lambda x: "|".join(x) if isinstance(x, list) else x
        )
        itf_save.to_csv(ITEM_FEATURES_FILE, index=False)
        logger.success(f"Sauvegardé : {ITEM_FEATURES_FILE}")
    return itf


# ── One-Hot Encoding des genres ───────────────────────────────────────────────
def encode_genres(movies: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """One-hot encoding des genres pour chaque film."""
    if save and GENRE_MATRIX_FILE.exists():
        logger.info(f"Genre matrix déjà présente : {GENRE_MATRIX_FILE}")
        return pd.read_csv(GENRE_MATRIX_FILE)

    genres_dummies = movies["genres_list"].explode()
    genres_dummies = pd.get_dummies(genres_dummies).groupby(level=0).max()
    result = movies[["movieid", "title_clean"]].join(genres_dummies)
    logger.info(f"Genre matrix : {result.shape} — {genres_dummies.shape[1]} genres")

    if save:
        result.to_csv(GENRE_MATRIX_FILE, index=False)
        logger.success(f"Sauvegardé : {GENRE_MATRIX_FILE}")
    return result


# ── Pipeline features ─────────────────────────────────────────────────────────
def run_feature_pipeline(ratings: pd.DataFrame,
                         movies: pd.DataFrame,
                         force: bool = False) -> dict:
    """Lance le pipeline features complet (interim → processed)."""
    if force:
        for f in [USER_ITEM_MATRIX_FILE, USER_FEATURES_FILE,
                  ITEM_FEATURES_FILE, GENRE_MATRIX_FILE]:
            f.unlink(missing_ok=True)

    surprise_ds  = build_surprise_dataset(ratings)
    user_matrix  = build_user_item_matrix(ratings)
    user_feats   = build_user_features(ratings)
    item_feats   = build_item_features(ratings, movies)
    genre_matrix = encode_genres(movies)

    logger.success("Feature engineering terminé (→ data/processed/)")
    return {
        "surprise_dataset": surprise_ds,
        "user_item_matrix": user_matrix,
        "user_features"   : user_feats,
        "item_features"   : item_feats,
        "genre_matrix"    : genre_matrix,
    }


if __name__ == "__main__":
    from data_loader import load_all
    data = load_all()
    run_feature_pipeline(data["ratings"], data["movies"])
